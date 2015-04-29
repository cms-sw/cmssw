#!/bin/bash

eval `scramv1 r -sh`

tnum=3
numev=3
step=a
DQMSEQUENCE=DQM

cmsDriver.py SingleMuPt10.cfi -s GEN,SIM,DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake -n ${numev} --eventcontent FEVTDEBUGHLT --datatier FEVTDEBUGHLT --conditions auto:run1_mc --mc --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p3.0.log

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_3_${step}_1}.xml


#cmsDriver.py step2_MC1_1 -s RAW2DIGI,RECO,VALIDATION:validation_preprod -n 3 --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier FEVTDEBUGHLT,DQMIO --conditions auto:run1_mc --mc --customise DQMServices/Components/test/customRecoSim.py >& p3.1.log 

#if [ $? -ne 0 ]; then
#	return 1
#fi

#cmsDriver.py step2_MC1_2 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n 3 --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:run1_mc --mc --customise DQMServices/Components/test/customRecoSim.py >& p3.2.log 

#if [ $? -ne 0 ]; then
#	return 1
#fi

#cmsDriver.py step2_MC1_3 -s RAW2DIGI,RECO,VALIDATION:validation_preprod,${DQMSEQUENCE} -n 3 --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:run1_mc --mc --customise DQMServices/Components/test/customRecoSim.py >& p3.3.log 

#if [ $? -ne 0 ]; then
#	return 1
#fi

#run first time
cmsDriver.py step2_MC1_4 -s RAW2DIGI,L1Reco,RECO,VALIDATION,${DQMSEQUENCE} -n ${numev} --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:run1_mc --mc --customise DQMServices/Components/test/customRecoSim.py --no_exec  --python_filename=test_${tnum}_${step}_2.py

cmsRun -e test_${tnum}_${step}_2.py >& p3.4.log
if [ $? -ne 0 ]; then
  return 1
fi

#run second time
cmsDriver.py step2_MC1_4 -s RAW2DIGI,L1Reco,RECO,VALIDATION,${DQMSEQUENCE} -n ${numev} --filein file:SingleMuPt10_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --eventcontent RECOSIM,DQM --datatier RECOSIM,DQMIO --conditions auto:run1_mc --mc --customise DQMServices/Components/test/customRecoSim.py --no_exec  --python_filename=test_${tnum}_${step}_2ndPass_2.py --fileout=secondPass.root

cmsRun -e test_${tnum}_${step}_2ndPass_2.py >& p3.4.2ndPass.log
if [ $? -ne 0 ]; then
  return 1
fi

#test merging
cp ../../merge.py ./

cmsRun -e merge.py inFiles="file:step2_MC1_4_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM_inDQM.root","file:secondPass_inDQM.root" >&  p3.merge.log
if [ $? -ne 0 ]; then
  return 1
fi



mv FrameworkJobReport{,_3_${step}_2}.xml
