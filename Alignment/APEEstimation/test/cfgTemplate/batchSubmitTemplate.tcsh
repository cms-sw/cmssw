#!/bin/tcsh


cd $CMSSW_BASE/src
#if [[ "$SHELL" == /bin/sh || "$SHELL" == /bin/bash || "$SHELL" == /bin/zsh ]] ; then
#  eval `scram runtime -sh`
#elif [[ "$SHELL" == /bin/csh || "$SHELL" == /bin/tcsh ]] ; then
eval `scram runtime -csh`
#else
#  echo "Unknown shell: $SHELL"
#  echo "cannot set CMSSW environment, stop processing"
#  exit 5
#fi
source /afs/cern.ch/cms/caf/setup.csh
cd -


cmsStageIn _THE_INPUTBASE__THE_NUMBER_.root reco.root


cmsRun $CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate/apeEstimator_cfg.py_THE_COMMANDS_






