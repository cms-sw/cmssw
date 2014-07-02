#!/bin/bash

eval `scramv1 r -sh`

tnum=1
DQMSEQUENCE=DQM
numev=500
step=a

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent RECO,DQM --datatier RECO,DQMIO --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__147647__0030E63A-1CD4-DF11-A1C2-0030487CD6B4.root --data --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& p1.1.log

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

cmsDriver.py test_${tnum}_${step}_2 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent RECO,DQM --datatier RECO,DQMIO --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__147647__02958305-11D4-DF11-A231-0030487CD77E.root --data --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=test_${tnum}_${step}_2.py 

cmsRun -e test_${tnum}_${step}_2.py >& p1.2.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml

cmsDriver.py test_${tnum}_${step}_3 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --eventcontent RECO,DQM --datatier RECO,DQMIO --conditions auto:com10  --filein file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Cosmics__RAW__v1__147647__029F4350-10D4-DF11-A37D-0030487CD6DA.root --data --customise DQMServices/Components/test/customReco.py --no_exec --python_filename=test_${tnum}_${step}_3.py

cmsRun -e test_${tnum}_${step}_3.py >& p1.3.log 

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_3}.xml
