#!/bin/bash

tnum=21
#rnum=173663
#rnum=175045
#rnum=173692
rnum=172822
step=c
eval `scramv1 r -sh`


cmsDriver.py test_${tnum}_${step}_1 -s HARVESTING:dqmHarvesting -n '-1' --conditions auto:com10 --datatier DQM --data --dbsquery="find file where dataset=/Jet/Run2011A-BoundaryTest*_520_12Mar2012-v1/DQM and site=srm-eoscms.cern.ch and run=${rnum}"  --scenario pp --customise_commands="process.dqmSaver.saveByLumiSection = cms.untracked.int32(1)\nprocess.dqmSaver.workflow = cms.untracked.string('/Jet/Run2011A-BoundaryTest_520_12Mar2012-v1/DQM')" --no_exec --python_filename=test_${tnum}_${step}_1.py

if [ -e  q${tnum}_c.1.log ]; then
  rm  q${tnum}_c.1.log
fi

cmsRun -e test_${tnum}_${step}_1.py >& q${tnum}_c.1.log

if [ $? -ne 0 ]; then
  return 1
fi

#mv FrameworkJobReport{,_11_b_1}.xml

#mv memory.out smemory_11.2.log
