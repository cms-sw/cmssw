#!/bin/bash

eval `scramv1 r -sh`

TNUM=1000
DQMSEQUENCE=DQM
NUMEV=1
STEP=a


cmsDriver.py Hydjet_Quenched_B8_2760GeV_cfi --no_exec --conditions auto:starthi_HIon --scenario HeavyIons -n ${NUMEV} --eventcontent FEVTDEBUGHLT -s GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,L1Reco,RECO,VALIDATION,${DQMSEQUENCE} --datatier GEN-SIM-DIGI-RECO --python_filename=test_${TNUM}_${STEP}_1.py --customise DQMServices/Components/test/customFEVTDEBUGHLT.py --fileout file:test_${TNUM}_${STEP}_1.root

cmsRun -e test_${TNUM}_${STEP}_1.py >& p${TNUM}.1.log

if [ $? -ne 0 ]; then
	exit 1
fi

mv FrameworkJobReport{,_${TNUM}_${STEP}_1}.xml
