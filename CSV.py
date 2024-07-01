import pandas as pd
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq




model = ChatGroq(
            model="llama3-70b-8192",groq_api_key="api_key", temperature= 0
        )




def analyst(input=""):
    model = ChatGroq(
            model="llama3-70b-8192",groq_api_key="api-key", temperature= 0
        )
    
    csv_path ="people.csv"
    agent = create_csv_agent(model, csv_path ,allow_dangerous_code=True)
    response = agent.run(input)
    return(response)
analyst_tool=Tool(
    name="Data_analyst",
    func=analyst,
    description=" use this tool to analyze a CSV dataset using a pre-trained large language mode.It interacts with the model to answer questions about the data and if need you can also use python language for task completion"
)



tools=[analyst_tool]
memory=ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True)


conversational_agent =initialize_agent(
agent='chat-conversational-react-description',
tools=tools,
llm=model,
verbose=True,
max_iterations=10000,
early_stopping_method='generate',
memory=memory,
handle_parsing_errors=True)

conversational_agent("""In the given csv file change the format of birth-date column format into yyyy/mm//dd and dont change 
                     remaining columns and save it as people.csv file
                     """)
                    