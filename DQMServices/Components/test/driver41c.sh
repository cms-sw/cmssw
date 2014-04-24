#!/bin/bash

tnum=41
#rnum=173663
#rnum=175045
#rnum=173692
#rnum=172822
step=c
DQMSEQUENCE=HARVESTING:dqmHarvesting
customFile=DQMServices/Components/test/test41_harvesting.py

eval `scramv1 r -sh`

for rnum in 172791 173241 173243 173244
  do
    # mv harvesting files for DQMIO step out of our way
    if [ -e DQM_V0001_R000${rnum}__Jet__Run2011A-BoundaryTest-v1__DQM.root ]; then
      mv DQM_V0001_R000${rnum}__Jet__Run2011A-BoundaryTest-v1__DQM.root  DQM_V0001_R000${rnum}__Jet__Run2011A-BoundaryTest-CJ__DQM.root
    fi
    cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} -n '-1' --conditions auto:com10 --datatier DQM --data --filein "file:file_${tnum}_a_old_Run${rnum}.root"  --scenario pp --customise ${customFile} --no_exec --python_filename=test_${tnum}_${step}_Run${rnum}.py

    if [ -e  q${tnum}_${step}_Run${rnum}.log ]; then
      rm  q${tnum}_${step}_Run${rnum}.log
    fi

    cmsRun -e test_${tnum}_${step}_Run${rnum}.py >& q${tnum}_${step}_Run${rnum}.log

    if [ $? -ne 0 ]; then
      return 1
    fi
  done

