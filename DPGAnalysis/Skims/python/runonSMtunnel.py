#!/usr/bin/env python
import os,string,sys,commands,time

# template file to be used
MODFILE="reco_skim_cfg_mod.py"

# the output file is PREFIX_date.root
PREFIX="SkimSM"

# it produces a file every NUEVENTS events
NUMEVENTS="-1"

while True:
    DATE=str(int(time.time()))
    print "Suffix:"+DATE
    FILENAME=PREFIX+"_"+DATE+"_cfg.py"
    FILELOG=PREFIX+"_"+DATE+".log"
    COMMAND="sed 's/SUFFIX/"+DATE+"/;s/NUMEVENTS/"+NUMEVENTS+"/' "+MODFILE+" > "+FILENAME
    os.system(COMMAND)
    print "Created: "+FILENAME+" . Running cmsRun now and logging in "+FILELOG
    os.system("cmsRun "+FILENAME+" 2>&1 | tee "+FILELOG+" | grep  --line-buffered -e \"Begin processing\" -e \"BeamSplash\"")
