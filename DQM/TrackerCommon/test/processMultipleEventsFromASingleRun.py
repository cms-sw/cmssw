import subprocess
import glob

#subprocess.Popen("ls", shell=True)

PD = "Jet"
print "PD = ", PD

runs=glob.glob(PD+"/*")

print "runs = ", runs

for run in runs: 
    events = glob.glob(PD+run)
    print "in run ", run ,"... the events are: ", events
    if len(events)>1 : print "MULTIPLE EVENTS IN ONE RUN... MERGE THEM!!!"
    else             : print "DONT PANIC, NOTHING!"


