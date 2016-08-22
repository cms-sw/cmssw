#!/bin/bash

numev=10
tnum=11
DQMSEQUENCE=DQM
step=a

eval `scramv1 r -sh`

#cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --datatier DQMIO --conditions auto:com10  --filein  file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/MinimumBias__RAW__v1__165633__1CC420EE-B686-E011-A788-0030487CD6E8.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_${tnum}_${step}_1.py
cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --datatier DQMIO --conditions 80X_dataRun2_Prompt_v9 --era Run2_2016 --filein /store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_${tnum}_${step}_1.py --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

