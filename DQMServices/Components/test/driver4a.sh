#!/bin/bash

eval `scramv1 r -sh`

numev=1000
tnum=4
step=a
DQMSEQUENCE=DQM

cmsDriver.py SingleMuPt10_cfi.py -s GEN,SIM,RECO,HLT:@fake,VALIDATION --fast --pileup=NoPileUp  --geometry DB --conditions=auto:run1_mc --eventcontent=FEVTDEBUGHLT,DQM --datatier FEVTDEBUGHLT,DQMIO -n ${numev} --no_exec --python_filename=test_${tnum}_${step}_1.py --customise DQMServices/Components/test/customFEVTDEBUGHLT.py  --fileout file:test_${tnum}_${step}_1.root

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml
