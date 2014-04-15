#!/bin/bash

eval `scramv1 r -sh`

tnum=3
numev=3
step=a
DQMSEQUENCE=DQM

cmsDriver.py SingleMuPt10.cfi -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,HLT:GRun -n ${numev} --eventcontent FEVTDEBUGHLT --datatier FEVTDEBUGHLT --conditions auto:startup_GRun --mc --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p3.0.log

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_3_${step}_1}.xml


#cmsDriver.py step2_MC1_1 -s RAW2DIGI,RECO,VALIDATION:validation_preprod -n 3 --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier FEVTDEBUGHLT,DQMIO --conditions auto:mc --mc --customise DQMServices/Components/test/customRecoSim.py >& p3.1.log 

#if [ $? -ne 0 ]; then
#	return 1
#fi

#cmsDriver.py step2_MC1_2 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n 3 --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:mc --mc --customise DQMServices/Components/test/customRecoSim.py >& p3.2.log 

#if [ $? -ne 0 ]; then
#	return 1
#fi

#cmsDriver.py step2_MC1_3 -s RAW2DIGI,RECO,VALIDATION:validation_preprod,${DQMSEQUENCE} -n 3 --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:mc --mc --customise DQMServices/Components/test/customRecoSim.py >& p3.3.log 

#if [ $? -ne 0 ]; then
#	return 1
#fi

cmsDriver.py step2_MC1_4 -s RAW2DIGI,RECO,VALIDATION,${DQMSEQUENCE} -n ${numev} --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:mc --mc --customise DQMServices/Components/test/customRecoSim.py --no_exec  --python_filename=test_${tnum}_${step}_2.py
#mv memory.out smemory_p.3.4.log
#mv checkMem.log p${tnum}.4.log

cmsRun -e test_${tnum}_${step}_2.py >& p3.4.log

if [ $? -ne 0 ]; then
  return 1
fi


mv FrameworkJobReport{,_3_${step}_2}.xml
