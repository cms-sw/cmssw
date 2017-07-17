#!/bin/bash

eval `scramv1 r -sh`

tnum=9
step=a
numev=1
DQMSEQUENCE=DQM

# GEN,FASTSIM,HLT:@fake,VALIDATION

cmd=`runTheMatrix.py --dryRun -l 5.1 -n -e --command "-n \${numev} --fileout file:test_\${tnum}_\${step}_1.root --no_exec --python_filename=test_\${tnum}_\${step}_1.py" | egrep "\[1\]" | egrep -o "cmsDriver.*"`

${cmd}

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.${step}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_a_2}.xml


