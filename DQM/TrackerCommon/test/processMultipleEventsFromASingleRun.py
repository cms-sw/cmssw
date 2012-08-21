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
    if len(events)<2 : print "DONT PANIC, JUST ONE EVENT IN THIS RUN!"
    files = ""
    for event in events:
        files+=" file:"+event+"/recoDQM_RAW2DIGI_RECO_DQM.root"
    print "files = ", files    
    if len(events)>1 : 
        print "MULTIPLE EVENTS IN ONE RUN... MERGE THEM!!!"
        newdir = run+"/MergedEvents"
        f1 = subprocess.Popen("mkdir "+newdir, shell=True)
        f1.wait()
        
        ### creating the configuration file to process DQM output file
        command2 = "cmsDriver.py offlineDQM -s HARVESTING:dqmHarvesting --conditions auto:com10 --data --filein "+files+" --scenario pp  --no_exec --python_filename=offlineDQM.py --dirout="+newdir
        f3 = subprocess.Popen(command2, shell=True)
        f3.wait()
        
        ### produce DQM file
        f4 = subprocess.Popen("cmsRun -e offlineDQM.py", shell=True)
        f4.wait()
        
                                                    
