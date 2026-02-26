#!/bin/bash

# input source
UNITFILE="/store/relval/CMSSW_16_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_Run4D110_PU-v1/2580000/025e04e8-4d2a-4d31-abb9-fb5f2280ea8f.root"
LOCALPATH='/eos/cms/store/relval/CMSSW_16_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_150X_mcRun4_realistic_v1_STD_Run4D110_PU-v1/2580000/'

if [[ "$1" == "unitTest" ]]; then
    echo "Running in UNIT TEST mode"
    ALL_FILES="$UNITFILE"
else
    echo "Input source: |${LOCALPATH}|"
    LOCALFILES=$(ls ${LOCALPATH})
    ALL_FILES=""
    for f in ${LOCALFILES[@]}; do
        ALL_FILES+="file:${LOCALPATH}/${f},"
    done
    ALL_FILES="${ALL_FILES%?}"  # drop last comma
fi

echo "Discovered files: $ALL_FILES"

## Step 2: reHLT, NANO, DQM
cmsDriver.py step2 -s L1P2GT,HLT:NGTScouting,NANO:@NGTScouting,DQM:@nanohltDQM \
	     --conditions auto:phase2_realistic_T35 \
	     --datatier DQMIO,NANOAODSIM \
	     -n 10 \
	     --eventcontent DQMIO,NANOAODSIM \
	     --geometry ExtendedRun4D110 \
	     --era Phase2C17I13M9 \
	     --procModifier alpaka,ngtScouting \
	     --filein $ALL_FILES \
	     --nThreads 24 \
	     --process HLTX \
	     --fileout file:step2.root \
	     --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
	     --python_filename step2.py

## Step 2: harvesting
cmsDriver.py step3 -s HARVESTING:@nanohltDQM \
	     --conditions auto:phase2_realistic_T35 \
	     --mc \
	     --geometry ExtendedRun4D110 \
	     --scenario pp \
	     --filetype DQM \
	     --era Phase2C17I13M9 \
	     -n 10000  \
	     --filein file:step2.root \
	     --fileout file:step3.root
