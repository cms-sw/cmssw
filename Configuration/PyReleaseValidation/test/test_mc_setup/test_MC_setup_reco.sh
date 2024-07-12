#!/bin/bash

############################################################################################################
### RECO + PAT

release=$1 
conditions=$2
era=$3

#export SCRAM_ARCH=$scram_arch

echo '> Running RAW2DIGI,L1Reco,RECO,RECOSIM + PAT steps in ' $release

cmsDriver.py  --python_filename reco.py --eventcontent AODSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier AODSIM --fileout file:step4.root \
--conditions $conditions --step RAW2DIGI,L1Reco,RECO,RECOSIM \
--geometry DB:Extended --filein file:step3.root --era $era \
--mc -n -1 --no_exec

if [ $? -ne 0 ]; then
    echo " !!!! Error in building the config for RECO with cmsDriver !!!!! "
    exit 1;
fi

cmsRun reco.py

if [ $? -ne 0 ]; then
    echo " !!!! Error in running the config for RECO !!!!! "
    exit 1;
fi


cmsDriver.py  --python_filename pat.py --eventcontent MINIAODSIM \
--customise Configuration/DataProcessing/Utils.addMonitoring \
--datatier MINIAODSIM --fileout file:step5.root \
--conditions $conditions --step PAT \
--geometry DB:Extended --filein file:step4.root --era $era \
--mc -n -1 --no_exec

if [ $? -ne 0 ]; then
    echo " !!!! Error in building the config for PAT with cmsDriver !!!!! "
    exit 1;
fi

cmsRun pat.py

if [ $? -ne 0 ]; then
    echo " !!!! Error in running the config for PAT !!!!! "
    exit 1;
fi


############################################################################################################
