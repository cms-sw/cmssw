#!/bin/bash

pwd=$PWD

# prompt analysis directory
export CAF_TRIGGER=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_TRIGGER
export L1CMSSW=CMSSW_4_2_3_patch2

# 64bit architecture
export SCRAM_ARCH=slc5_amd64_gcc434

# GRID UI
source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh

# setup CMSSW environment
cd $CAF_TRIGGER/l1analysis/cmssw/$L1CMSSW/src
eval `scram ru -sh`
cd $pwd

# CRAB
#source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh
source /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh

# include scripts in path
export PATH=${PATH}:$CAF_TRIGGER/econte
export CASTORPATH=rfio:///castor/cern.ch/cms/store/caf/user/L1AnalysisNtuples/

# add jobs to pythonpath
export PYTHONPATH=${PYTHONPATH}:$CAF_TRIGGER/econte/$L1CMSSW/src/UserCode/L1TriggerDPG/test

# add ROOT macros to ROOTPATH - doesn't work, grrr
export ROOTPATH=${ROOTPATH}:$CAF_TRIGGER/l1analysis/macros

if [ ! -e "$CAF_TRIGGER/$USER/ntuples" ] 
    then ln -s $CAF_TRIGGER/l1analysis/ntuples $CAF_TRIGGER/$USER/ntuples
fi
if [ ! -e "$CAF_TRIGGER/$USER/macros" ] 
    then ln -s $CAF_TRIGGER/l1analysis/macros $CAF_TRIGGER/$USER/macros
fi

