#!/bin/bash

eval `scramv1 r -sh`

numev=-1
tnum=21
step=a
DQMSEQUENCE=DQM

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --processName=RERECO --eventcontent DQM --datatier DQMIO --conditions auto:com10 --data --dbsquery='find file where dataset=/Jet/Run2011A-BoundaryTest-HighMET-10Oct2011-44-v1/RAW and site=srm-eoscms.cern.ch' --customise_commands="process.DQMStore.forceResetOnBeginRun = cms.untracked.bool(True)" --python_filename=test_${tnum}_${step}_1.py --no_exec

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1}.xml

