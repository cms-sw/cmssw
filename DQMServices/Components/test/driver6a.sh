#!/bin/bash

eval `scramv1 r -sh`

tnum=6
numev=10
step=a

cmsDriver.py MinBias.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:@fake -n ${numev} --eventcontent FEVTDEBUGHLT,DQM --datatier FEVTDEBUGHLT,DQMIO --conditions auto:run1_mc --customise=SimCalorimetry/HcalZeroSuppressionProducers/NoHcalZeroSuppression_cff.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p6.0.log

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

cmsDriver.py test_${tnum}_${step}_2 -s ALCA:HcalCalMinBias -n ${numev} --eventcontent ALCARECO,DQM --datatier FEVTDEBUGHLT,DQMIO --conditions auto:run1_mc --filein file:MinBias_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --mc --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_2.py

cmsRun -e test_${tnum}_${step}_2.py >& p6.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml
