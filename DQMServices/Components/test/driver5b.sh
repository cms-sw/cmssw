#!/bin/bash

eval `scramv1 r -sh`

tnum=5
step=b

cmsDriver.py test_${tnum}_${step}_1 -s ALCA:TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+EcalCalElectron+HcalCalIsoTrk+HcalCalDijets+HcalCalHO+MuAlOverlaps -n 1000 --conditions auto:mc --eventcontent ALCARECO --filein file:test_${tnum}_a_2_RAW2DIGI_RECO.root --customise DQMServices/Components/test/customHarvesting.py --no_exec --python_filename=test_${tnum}_${step}_1.py

cmsRun -e test_${tnum}_${step}_1.py >& q5.1.log

if [ $? -ne 0 ]; then
	return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml
