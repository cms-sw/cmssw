#!/bin/bash

eval `scramv1 r -sh`

tnum=110
DQMSEQUENCE=HARVESTING:dqmHarvesting
step=b

#python /afs/cern.ch/user/r/rovere/public/checkMem.py cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions auto:com10 --data --filein file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root --scenario pp --customise DQMServices/Components/test/customHarvestingCJ.py --no_exec --python_filename=test_${tnum}_${step}_1.py
cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions auto:com10 --data --filein file:test_${tnum}_a_1_RAW2DIGI_RECO_DQM.root --scenario pp --customise DQMServices/Components/test/customHarvestingCJ.py --no_exec --python_filename=test_${tnum}_${step}_1.py

sed -i -e "s/PoolSource/DQMRootSource/" test_${tnum}_${step}_1.py
sed -i -e "s/.*processingMode.*//" test_${tnum}_${step}_1.py
cmsRun -e test_${tnum}_${step}_1.py >& q${tnum}.1.log

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

#mv memory.out smemory_${tnum}.2.log
#mv checkMem.log q${tnum}.1.log
