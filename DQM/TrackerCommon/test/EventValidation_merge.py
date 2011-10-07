import re
import subprocess

file = open('interestingEvents_Mu.txt', 'r')
events = file.readlines()

runnumber_interest = {}
eventnumber_interest = {}

command = "hadd Zmumubb_DQM.root "

for i in range (1,len(events)): 

    runinfo_interest  = re.split('[:]',str(events[i]))

    print "runinfo interest  = ", runinfo_interest

    runnumber_interest[i]   = runinfo_interest[0]
    eventnumber_interest[i] = runinfo_interest[2][:-1]

    print "run nr   = ", runnumber_interest[i]
    print "event nr = ", eventnumber_interest[i]
    
    command += runnumber_interest[i]+"_"+eventnumber_interest[i]+".root "

print "command = ", command

subprocess.Popen(command, shell=True)

