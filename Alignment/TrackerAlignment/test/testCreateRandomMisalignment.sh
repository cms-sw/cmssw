#! /bin/bash

folder=$CMSSW_BASE/src/Alignment/TrackerAlignment/test/Misalignments

for a in $(ls $folder); do
    echo "running " ${a}
    cmsRun $folder/${a}
done 
