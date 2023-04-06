#!/bin/bash

function die { echo $1: status $2 ; exit $2; }
folder=$CMSSW_BASE/src/Alignment/TrackerAlignment/test/Misalignments

for a in $(ls $folder); do
    echo "running unit test: " ${a}
    cmsRun $folder/${a} || die "Failure running ${a}" $?
done 
