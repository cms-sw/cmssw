#! /usr/bin/env python
#Quick and dirty script to provide the necessary (ordered by timestamp list of logs in the) list.txt file used by the ExtractTrends.C root macro
import os
import time

ls=os.listdir(os.getcwd())
TimesLogs=[]
for log in ls:
    if "DetVOffReaderSummary__FROM" in log:
        (start,end)=log[:-4].split("FROM_")[1].split("_TO_")
        CurrentTime=time.mktime(time.strptime(start.replace("__","_0"),"%a_%b_%d_%H_%M_%S_%Y"))
        TimesLogs.append((CurrentTime,log))
TimesLogs.sort()
listfile=open("list.txt","w")
for log in TimesLogs:
    print log[1]
    listfile.write(log[1]+"\n")
