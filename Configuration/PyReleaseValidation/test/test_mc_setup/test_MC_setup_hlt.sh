#!/bin/bash

############################################################################################################
### HLT STEP in a different release

conditions=$1
era=$2
hlt=$3

echo ">>> Running HLT:${hlt} step in " $CMSSW_BASE

config_name=hlt_"$(echo "$hlt" | tr ':' _ | tr '@' _ )".py
cmsDriver.py  --python_filename $config_name --eventcontent RAWSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier GEN-SIM-RAW --fileout file:step3.root \
--conditions $conditions \
--customise_commands 'process.source.bypassVersionCheck = cms.untracked.bool(True)' \
--step 'HLT:'$hlt --geometry DB:Extended --filein file:step2.root --era $era --mc -n -1 --no_exec

if [ $? -ne 0 ]; then
    exit 1;
fi

cmsRun $config_name

if [ $? -ne 0 ]; then
    exit 1;
fi

############################################################################################################
