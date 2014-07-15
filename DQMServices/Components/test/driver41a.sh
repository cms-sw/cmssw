#!/bin/bash

eval `scramv1 r -sh`

numev=-1
tnum=41
step=a
DQMSEQUENCE=DQM

# original Custom File
customFile=DQMServices/Components/test/test41.py

cmsDriver.py test_${tnum}_${step}_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --processName=RERECO --eventcontent DQMIO --datatier DQMIO --conditions auto:com10 --data --filein 'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Skim_Run172791_173241_173243_173244.root' --customise ${customFile} --python_filename=test_${tnum}_${step}_new_Run172791_173241_173243_173244.py --no_exec --fileout file_${tnum}_${step}_new_Run172791_173241_173243_173244.root

cmsRun -e test_${tnum}_${step}_new_Run172791_173241_173243_173244.py >& test_${tnum}_${step}_new_Run172791_173241_173243_173244.log

if [ $? -ne 0 ]; then
  echo "test_${tnum}_${step}_new_Run172791_173241_173243_173244.py failed."
  return 1
fi

for rnum in 172791 173241 173243 173244
do
  # mv the eventual FirstStep file produced in the previous step out of our way
  if [ -e DQM_V0001_R000${rnum}__Jet__Run2011A-BoundaryTest-v1-FirstStep__DQM.root ]; then
    mv DQM_V0001_R000${rnum}__Jet__Run2011A-BoundaryTest-v1-FirstStep__DQM.root DQM_V0001_R000${rnum}__Jet__Run2011A-BoundaryTest-v1-FirstStep-CJ__DQM.root
  fi
  
  cmsDriver.py test_${tnum}_a1_1 -s RAW2DIGI,RECO,${DQMSEQUENCE} -n ${numev} --processName=RERECO --eventcontent DQM --datatier DQM --conditions auto:com10 --data --filein 'file:/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/DQMTest/Skim_Run'${rnum}'.root' --customise ${customFile} --python_filename=test_${tnum}_${step}_old_Run${rnum}.py --no_exec --fileout file_${tnum}_${step}_old_Run${rnum}.root

  cmsRun -e test_${tnum}_${step}_old_Run${rnum}.py >& test_${tnum}_${step}_old_Run${rnum}.log

  if [ $? -ne 0 ]; then
    echo "test_${tnum}_${step}_old_Run${rnum}.py failed."
    return 1
  fi
done

