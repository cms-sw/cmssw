#!/bin/bash

DQMSEQUENCE=DQM
tnum=110
numev=100
step=a

eval `scramv1 r -sh`

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent DQM --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__147647__0030E63A-1CD4-DF11-A1C2-0030487CD6B4.root --data --customise DQMServices/Components/test/customDQMCJ.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv testCJones_RunLumi.root test_${tnum}_${step}_1_RAW2DIGI_RECO_DQM.root
mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

