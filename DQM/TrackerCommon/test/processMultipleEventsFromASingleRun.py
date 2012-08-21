import subprocess
import glob

#subprocess.Popen("ls", shell=True)

PD = "Jet"
print "PD = ", PD

dir1 = PD+"/*"
print "dir1 = ", dir1
runs=glob.glob(dir1)

print "runs = ", runs

for run in runs: 
    dir2 = run+"/*"
    print "dir2 = ", dir2
    events = glob.glob(dir2)
    print "in run ", run ,"... the events are: ", events
    if len(events)>1 : print "MULTIPLE EVENTS IN ONE RUN... MERGE THEM!!!"
    else             : print "DONT PANIC, NOTHING!"


