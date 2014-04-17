#!/bin/bash

tnum=41
#rnum=173663
#rnum=175045
#rnum=173692
#rnum=172791
step=b
eval `scramv1 r -sh`
DQMSEQUENCE=HARVESTING:dqmHarvesting

customFile=DQMServices/Components/test/test41_harvesting.py

for rnum in 172791 173241 173243 173244
do
  cmsRun ../../filterByRun2.py ${rnum}

  cmsDriver.py test_${tnum}_${step}_1 -s ${DQMSEQUENCE} --conditions auto:com10 --datatier DQMIO --filetype DQM --data --filein "file:file_${tnum}_a_new_Run${rnum}.root"  --scenario pp --customise ${customFile} --no_exec --python_filename=test_${tnum}_${step}_new_Run${rnum}.py

  if [ -e  q${tnum}_${step}_new_Run${rnum}.log ]; then
    rm  q${tnum}_${step}_new_Run${rnum}.log
  fi

  cmsRun -e test_${tnum}_${step}_new_Run${rnum}.py >& q${tnum}_${step}_new_Run${rnum}.log

  if [ $? -ne 0 ]; then
    echo "test_${tnum}_${step}_new_Run${rnum}.py failed."
    return 1
  fi
done

