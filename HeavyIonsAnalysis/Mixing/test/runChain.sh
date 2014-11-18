#!/bin/sh

for i in 1 2 3
do
    cmsRun workflowD_step${i}.py > step${i}.out 2> step${i}.err
done

cmsRun validateHiMixing.py > mix.out 2> mix.err

