import pickle

f=open('RunPromptRecoCfg.pkl','rb')
process = pickle.load(f)
fout=open("config_dump.py", "w")
fout.write(process.dumpPython())
