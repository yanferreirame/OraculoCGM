from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os 
import pandas as pd
import streamlit as st

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")



st.set_page_config(page_title="Oráculo da Controladoria", page_icon="🧙‍♂️")
st.title("🧙‍♂️ Oráculo da Controladoria")



def setup_chain():
    loader = CSVLoader(file_path= "basecontroladoria.csv")
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI()

    rag_template = """
    Voce é oráculo.
    Seu trabalho é fornecer informações sobre a Controladoria,
    respondendo com base no banco de dados fornecido como contexto.

    Contexto: {context}

    Pergunta do Cliente: {question}
    """

    prompt = ChatPromptTemplate.from_template(rag_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain

chain = setup_chain()
user_question = st.text_input("Faça sua pergunta ao Oráculo:")

if user_question:
    response = chain.invoke(user_question)
    st.markdown(f"**Resposta:** {response.content}")

    