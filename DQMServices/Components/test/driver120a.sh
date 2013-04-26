#!/bin/bash

eval `scramv1 r -sh`

tnum=120
numev=100
DQMSEQUENCE=DQM
step=a

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --conditions auto:com10 --scenario HeavyIons  --data --customise DQMServices/Components/test/customHeavyIonsCJ.py --processName RECO2 --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv testCJones_RunLumi.root test_${tnum}_${step}_1_RAW2DIGI_RECO_DQM.root
mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

