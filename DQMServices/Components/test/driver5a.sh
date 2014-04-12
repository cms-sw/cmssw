#!/bin/bash

eval `scramv1 r -sh`

tnum=5
numev=10
step=a

cmsDriver.py TTbar_Tauola.cfi -s GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun -n ${numev} --conditions auto:startup_GRun --relval 9000,50 --datatier 'GEN-SIM-RAW' --eventcontent RAWSIM --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p5.0.log

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

cmsDriver.py test_${tnum}_${step}_2 -s RAW2DIGI,RECO -n ${numev} --eventcontent RECOSIM --conditions auto:startup_GRun --filein file:TTbar_Tauola_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_HLT.root --no_exec --python_filename=test_${tnum}_${step}_2.py

cmsRun -e test_${tnum}_${step}_2.py >& p5.1.log 
if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml
