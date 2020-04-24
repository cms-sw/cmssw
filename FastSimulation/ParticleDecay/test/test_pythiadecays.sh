#!/bin/bash

pids=('15' '211' '111' '130' '411' '521')
nevents=1000
for pid in ${pids[@]}
do
    echo "# create event generation cfg for pid $pid"
    source createGenSimCfg.sh $pid $nevents
    echo "# run event generation for pid $pid"
    cmsRun pgun_${pid}.py &> pgun_${pid}.stdout.txt
    echo "# analyse events for pid $pid"
    cmsRun ../python/TestPythiaDecays_cfg.py inputFiles="file:pgun_${pid}.root" outputFile="file:pgun_${pid}_test.root" &> pgun_${pid}_test.stdout.txt
    echo "# draw validation plots for pid $pid"
    python drawComparison.py pgun_${pid}_test.root figures/pgun_${pid}
done
