#!/bin/bash

############################################################################################################
### GEN-SIM-RAW-DIGI

release=$1 
conditions=$2
era=$3
beamspot=$4

echo '> Running GEN,SIM + DIGI,L1,DIGI2RAW steps in ' $release

cmsDriver.py TTbar_14TeV_TuneCP5_cfi --python_filename gen_sim.py \
--eventcontent RAWSIM --customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier GEN-SIM --fileout file:step1.root --conditions $conditions --beamspot $beamspot \
--step GEN,SIM --geometry DB:Extended --era $era --mc -n 10 --no_exec

if [ $? -ne 0 ]; then
    echo " !!!! Error in building the config for GEN-SIM with cmsDriver !!!!! "
    exit 1;
fi

cmsRun gen_sim.py

if [ $? -ne 0 ]; then
    echo " !!!! Error in running the config for GEN-SIM !!!!! "
    exit 1;
fi

cmsDriver.py  --python_filename raw_digi.py --eventcontent RAWSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier GEN-SIM-RAW --filein file:step1.root \
--fileout file:step2.root --conditions $conditions --step DIGI,L1,DIGI2RAW \
--geometry DB:Extended --era $era --mc -n -1 --no_exec

if [ $? -ne 0 ]; then
    echo " !!!! Error in building the config for RAW-DIGI with cmsDriver !!!!! "
    exit 1;
fi

cmsRun raw_digi.py

if [ $? -ne 0 ]; then
    echo " !!!! Error in running the config for RAW-DIGI !!!!! "
    exit 1;
fi
############################################################################################################
