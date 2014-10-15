#!/bin/bash

eval `scramv1 r -sh`

tnum=18
numev=10
DQMSEQUENCE=DQM

cmsDriver.py SingleMuPt10.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:@fake -n ${numev} --eventcontent FEVTDEBUGHLT --conditions auto:mc --mc --no_exec --python_filename=test_${tnum}.py

cmsRun -e test_${tnum}.py >& p${tnum}.0.log

cmsDriver.py step_VALIDATION -s RAW2DIGI,RECO,VALIDATION -n ${numev} --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM --conditions auto:mc --mc --customise DQMServices/Components/test/customRecoSimIgProf.py --no_exec --python_filename test_validation_cfg.py

igprof -d -t cmsRun -mp -z -o IgProfCumulative_10.gz cmsRun test_validation_cfg.py &> p${tnum}.1.log

if [ $? -ne 0 ]; then
  return 1
fi
