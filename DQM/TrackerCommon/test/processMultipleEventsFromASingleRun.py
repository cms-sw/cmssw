###########################################################################
###########################################################################
###                                                                     ###   
### This is the amazing script to                                       ###
### process multiple RECO events from the same run                      ###
### into a single DQM file which can be uploaded in the GUI             ###
###                                                                     ###   
### This can be used for RECO files created with getEventsAndProcess.py ###
###                                                                     ###   
### ...but it is still written such that many directory structure like  ###
### <PD>/<run>/<event>/<filename>                                       ### 
### is recongized                                                       ###
###                                                                     ###   
### Functionality: get RECO files and make a single DQM file            ###
###                                                                     ###   
### To be done by user                                                  ###   
### 0) Run python -i getEventsAndProcess.py                             ###   
### 1) Specify PD                                                       ###   
### 2) Run python -i processMultipleEventsFromASingleRun.py             ###   
###                                                                     ###
### To do:                                                              ###
### - repair outdir option when running cmsDriver                       ###
### - make more elegant how to create/remove merge directory            ###
###                                                                     ###
### Doei!                                                               ###
###                                                                     ###
###########################################################################
###########################################################################

import subprocess
import glob

### Specify PD ###

PD = "Jet"
print "PD = ", PD

### Runs in PD

dir1 = PD+"/*"
print "dir1 = ", dir1
runs=glob.glob(dir1)
print "runs = ", runs

for run in runs:
    
    ### specify directory
    newdir = run+"/MergedEvents"
    "directory to merge = ", newdir
    f1 = subprocess.Popen("rm -rf "+newdir, shell=True)
    f1.wait()

    ### get event(s) from run directory
    dir2 = run+"/*"
    print "dir2 = ", dir2
    events = glob.glob(dir2)
    print "in run ", run ,"... the events are: ", events
    if len(events)<2 :
        print "===> DONT PANIC, JUST ONE EVENT IN ", run
    else :
        print "==> MERGE THE MULTIPLE EVENTS IN ", run

        ### specify input files
        files = ""
        for event in events: files+="file:"+event+"/recoDQM_RAW2DIGI_RECO_DQM.root,"
        print "files = ", files

        ### make merge dir
        f2 = subprocess.Popen("mkdir "+newdir, shell=True)
        f2.wait()

        ### create the configuration file to process DQM output file
        command2 = "cmsDriver.py offlineDQM -s HARVESTING:dqmHarvesting --conditions auto:com10 --data --filein "+files[:-1]+" --scenario pp  --no_exec --python_filename=offlineDQM.py --dirout=dir:"+newdir
        f3 = subprocess.Popen(command2, shell=True)
        f3.wait()
        
        ### produce DQM file
        f4 = subprocess.Popen("cmsRun -e offlineDQM.py", shell=True)
        f4.wait()
        
        ### move DQM file to new dir
        ### This is an (ugly) fix since cmsDriver doesnt pick up the "dirout" command
        mvfile="DQM_V0001_R000"+run[-6:]+"__Global__CMSSW_X_Y_Z__RECO.root"
        print "output dir = ", newdir
        mvcommand = "mv "+mvfile+" "+newdir+"/."
        print "command to move = ", mvcommand
        f5 = subprocess.Popen(mvcommand, shell=True)
        f5.wait()

print "===> SINGLE DQM FILES CAN BE FOUND IN <PD>/<RUN>/<EVENT>"
print "===> MERGED DQM FILES CAN BE FOUND IN <PD>/<RUN>/MergedEvents"
                                    
