#!/bin/bash

eval `scramv1 r -sh`

step=a
tnum=8
numev=1
DQMSEQUENCE=DQM


# GEN-SIM
cmd=`runTheMatrix.py --dryRun -l 201 -n -e --command "-n \${numev} --fileout file:test_\${tnum}_\${step}_1.root --no_exec --python_filename=test_\${tnum}_\${step}_1.py" | egrep "\[1\]" | egrep -o "cmsDriver.*"`

${cmd}

cmsRun -e test_${tnum}_${step}_1.py >& p${tnum}.${step}.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

# DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,L1Reco

cmd=`runTheMatrix.py --dryRun -l 201 -n -e --command "-n \${tnum} --fileout file:test_\${tnum}_\${step}_1_1.root --no_exec --python_filename=test_\${tnum}_\${step}_1_1.py --filein file:test_\${tnum}_\${step}_1.root" | egrep "\[2\]" | egrep -o "cmsDriver.*"`

${cmd}

cmsRun -e test_${tnum}_${step}_1_1.py >& p${tnum}.${step}.1.1.log 

if [ $? -ne 0 ]; then
  return 1
fi

mv FrameworkJobReport{,_${tnum}_${step}_1.1}.xml

