#!/usr/bin/env python
import os,string,sys,commands,time

# template file to be used
MODFILE="reco_skim_cfg_mod.py"

# the output file is PREFIX_date.root
PREFIX="SkimSM"

# it produces a file every NUEVENTS events
NUMEVENTS="-1"

# here starts the main

if len(sys.argv)!=2 :
    print "Usage = runonSM.py <type>"
    print "where type is either \"tunnel\" or \"revproxy\" or \"playback\" "
    sys.exit(1)

TYPE=sys.argv[1]

if TYPE=="tunnel" :
    SOURCE="cms.string('http://localhost:22100/urn:xdaq-application:lid=30')"
    SELECTHLT= "cms.untracked.string('hltOutputDQM')"
elif TYPE=="revproxy":
    SOURCE="cms.string('http://cmsdaq0.cern.ch/event-server/urn:xdaq-application:lid=30')"
    SELECTHLT= "cms.untracked.string('hltOutputDQM')"
elif TYPE=="playback":
    SOURCE="cms.string('http://localhost:50082/urn:xdaq-application:lid=29')"
    SELECTHLT= "cms.untracked.string('hltOutputDQM')"
else:
    print "wrong type value."
    sys.exit(1)
    
while True:
    DATE=str(int(time.time()))
    print "Suffix:"+DATE
    FILENAME=PREFIX+"_"+DATE+"_cfg.py"
    FILELOG=PREFIX+"_"+DATE+".log"
    # read mod file
    modfile=open(MODFILE,"r")
    text=modfile.read()
    modfile.close()
    text=text.replace("SUFFIX",DATE)
    text=text.replace("SOURCE",SOURCE)
    text=text.replace("NUMEVENTS",NUMEVENTS)
    text=text.replace("SELECTHLT",SELECTHLT)
    newfile=open(FILENAME,"w")
    newfile.write(text)
    newfile.close()
                 
    print "Created: "+FILENAME+" . Running cmsRun now and logging in "+FILELOG
    os.system("cmsRun "+FILENAME+" 2>&1 | tee "+FILELOG+" | grep  --line-buffered -e \"Begin processing\" -e \"BeamSplash\" -e \"PhysDecl\"")
