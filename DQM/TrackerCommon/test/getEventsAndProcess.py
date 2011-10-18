import os
import subprocess
import sys
import re
import time

file = open('interestingEvents_El.txt', 'r')
##firstline could be primary dataset.
myPD = file.readline() ##DoubleMu
#rd   = subprocess.Popen("rm -rf FinalOutput", shell=True)
#rd.wait()
#nd   = subprocess.Popen("mkdir FinalOutput", shell=True)
#nd.wait()

events = file.readlines()

for i in range(0,len(events)):
    print "################################################################################################"
    print "***> Interesting event #", i , " : " + events[i]
    pickEvents = "edmPickEvents.py *"+ str(myPD[:-1])+"*RAW " + str(events[i][:-1])
    print "***> To pick events :", pickEvents
    pickEventOutput = os.popen(pickEvents).readlines()

    runinfo = re.split('[:]',str(events[i]))
    pickEventOutput[1] = pickEventOutput[1][:-9]+"_run"+runinfo[0]+"_event"+runinfo[2][:-1]+".root "
    print "***> To pick and copy :", pickEventOutput[1]
    prestage = "stager_get -M /castor/cern.ch/cms/"+(pickEventOutput[3])[14:]
    print "prestage = ",  prestage 
    os.popen(prestage)

for i in range(0,len(events)):
    print "################################################################################################"
    print "***> Interesting event #", i , " : " + events[i]
    pickEvents = "edmPickEvents.py *"+ str(myPD[:-1])+"*RAW " + str(events[i][:-1])
    print "***> To pick events :", pickEvents
    pickEventOutput = os.popen(pickEvents).readlines()

    runinfo = re.split('[:]',str(events[i]))
    pickEventOutput[1] = pickEventOutput[1][:-9]+"_run"+runinfo[0]+"_event"+runinfo[2][:-1]+".root "
    print "***> To pick and copy :", pickEventOutput[1]
    edmCopyPickMerge=pickEventOutput[1]+pickEventOutput[2][:-2]+" "+pickEventOutput[3]
    print "edmCopyPickMerge = ",  edmCopyPickMerge 
    os.popen(edmCopyPickMerge)
        
    runDQM = str("./EventValidation_runofflineDQM.sh file:./"+pickEventOutput[1][28:-1])
    print "***> To process events RAW->DQM : ", runDQM

    command1 = "cmsDriver.py recoDQM -s RAW2DIGI,RECO,DQM --eventcontent DQM --conditions auto:com10 --geometry Ideal --filein file:./"+pickEventOutput[1][28:-1]+" --data --no_exec --python_filename=recoDQM.py" 

    print "command1 = " , command1
    f = subprocess.Popen(command1, shell=True)
    f.wait()

    f2 = subprocess.Popen("cmsRun -e recoDQM.py", shell=True)
    f2.wait()

    nd2 = subprocess.Popen("mkdir ./FinalOutput/Event"+runinfo[2][:-1], shell=True)
    # creating the configuration file to process DQM output file
    command2 = "cmsDriver.py offlineDQM -s HARVESTING:dqmHarvesting --conditions auto:com10 --data --filein file:recoDQM_RAW2DIGI_RECO_DQM.root --scenario pp  --no_exec --python_filename=offlineDQM.py --dirout=./FinalOutput/Event"+runinfo[2][:-1]
    f3 = subprocess.Popen(command2, shell=True)
    f3.wait()

    f4 = subprocess.Popen("cmsRun -e offlineDQM.py", shell=True)
    f4.wait()

    outputdir="./FinalOutput/Event"+runinfo[2][:-1]
    mvfile="DQM_V0001_R000"+runinfo[0]+"__Global__CMSSW_X_Y_Z__RECO.root"
    
    print "output dir = ", outputdir

    mvcommand = "mv "+mvfile+" "+outputdir+"/."
    print "command to move = ", mvcommand

    f5 = subprocess.Popen(mvcommand, shell=True)
    f5.wait()
