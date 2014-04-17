#!/bin/bash

eval `scramv1 r -sh`

step=b
tnum=8
numev=1
DQMSEQUENCE=VALIDATION,DQM

cmd=`runTheMatrix.py --dryRun -l 201 -n -e -i all --command " -n \${numev} --datatier DQMIO --eventcontent DQM --python_filename=test_\${tnum}_\${step}_1.py --filein file:test_\${tnum}_a_1_1.root --fileout file:test_\${tnum}_\${step}_1.root --no_exec --customise DQMServices/Components/test/customDQM.py" | egrep "\[3\]" | egrep -o "cmsDriver.*"`

${cmd}

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.${step}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_2}.xml
