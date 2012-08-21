import os
import subprocess
import sys
import re
import time

file = open('interestingEvents_MET.txt', 'r')
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

    ### define new dir: <PD>/<run>/Event"+<eventnr>
    newdir1 = str(myPD[:-1])
    newdir2 = runinfo[0]
    newdir3 = "Event"+runinfo[2][:-1]
    newdir = "./"+newdir1+"/"+newdir2+"/"+newdir3
    print "***> Ouput directory for this event = ", newdir

    ### pick and copy
    pickEventOutput[1] = pickEventOutput[1][:-9]+"_run"+runinfo[0]+"_event"+runinfo[2][:-1]+".root "
    print "***> To pick and copy :", pickEventOutput[1]
    edmCopyPickMerge=pickEventOutput[1]+pickEventOutput[2][:-2]+" "+pickEventOutput[3]
    print "edmCopyPickMerge = ",  edmCopyPickMerge 
    os.popen(edmCopyPickMerge)

    ### raw -> dqm 
    x = pickEventOutput[1][28:-1]
    runDQM = str("./EventValidation_runofflineDQM.sh file:./"+x)
    print "***> To process events RAW->DQM : ", runDQM

    ### cmsDriver command
    command1 = "cmsDriver.py recoDQM -s RAW2DIGI,RECO,DQM --eventcontent DQM --conditions auto:com10 --geometry Ideal --filein file:./"+x+" --data --no_exec --python_filename=recoDQM.py" 
    print "command1 = " , command1
    f = subprocess.Popen(command1, shell=True)
    f.wait()

    ### cmsRun
    f2 = subprocess.Popen("cmsRun -e recoDQM.py", shell=True)
    f2.wait()

    ### make dir and move reco file there
    nd2 = subprocess.Popen("mkdir "+newdir1, shell=True)
    print "mkdir ", newdir1
    nd4 = subprocess.Popen("mkdir "+newdir1+"/"+newdir2, shell=True)
    print "mkdir ", newdir1+"/"+newdir2
    nd5 = subprocess.Popen("mkdir "+newdir1+"/"+newdir2+"/"+newdir3, shell=True)
    print "mkdir ", newdir1+"/"+newdir2+"/"+newdir3

    print newdir 

    print "copying to new dir"
    nd3 = subprocess.Popen("cp recoDQM_RAW2DIGI_RECO_DQM.root "+newdir+"/.", shell=True)

    #nd2 = subprocess.Popen("mkdir ./FinalOutput/Event"+runinfo[2][:-1], shell=True)

    ### creating the configuration file to process DQM output file
    command2 = "cmsDriver.py offlineDQM -s HARVESTING:dqmHarvesting --conditions auto:com10 --data --filein file:recoDQM_RAW2DIGI_RECO_DQM.root --scenario pp  --no_exec --python_filename=offlineDQM.py --dirout="+newdir
    f3 = subprocess.Popen(command2, shell=True)
    f3.wait()

    ### produce DQM file
    f4 = subprocess.Popen("cmsRun -e offlineDQM.py", shell=True)
    f4.wait()

    ### move DQM file to new dir
    outputdir="./FinalOutput/Event"+runinfo[2][:-1]
    mvfile="DQM_V0001_R000"+runinfo[0]+"__Global__CMSSW_X_Y_Z__RECO.root"
    print "output dir = ", outputdir
    print "new dir = ", newdir
    mvcommand = "mv "+mvfile+" "+newdir+"/."
    print "command to move = ", mvcommand
    f5 = subprocess.Popen(mvcommand, shell=True)
    f5.wait()
