#!/bin/bash

tnum=21
#rnum=173663
#rnum=175045
#rnum=173692
rnum=172822
step=b
eval `scramv1 r -sh`

cmsRun ../../filterByRun.py

cmsDriver.py test_${tnum}_${step}_1 -s HARVESTING:dqmHarvesting --conditions auto:com10 --datatier DQMIO --filetype DQM --data --filein "file:test_${tnum}_a_1_${rnum}_RAW2DIGI_RECO_DQM.root"  --scenario pp --customise_commands="process.dqmSaver.saveByLumiSection = cms.untracked.int32(1)\nprocess.dqmSaver.workflow = cms.untracked.string('/Jet/Run2011A-BoundaryTest-HighMET-06Oct2011-44-v1CJ/DQM')" --no_exec --python_filename=test_${tnum}_${step}_1.py

if [ -e  q${tnum}_${step}.1.log ]; then
  rm  q${tnum}_${step}.1.log
fi

cmsRun -e test_${tnum}_${step}_1.py >& q${tnum}_${step}.1.log

if [ $? -ne 0 ]; then
  return 1
fi

#mv FrameworkJobReport{,_11_b_1}.xml

#mv memory.out smemory_11.2.log
