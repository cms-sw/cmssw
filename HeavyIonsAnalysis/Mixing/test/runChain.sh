#!/bin/sh

for i in 3
do
    cmsRun workflowD_step${i}.py > step${i}.out 2> step${i}.err
done

cmsRun validateHiMixing.py > mix.out 2> mix.err

