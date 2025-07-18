#!/bin/bash

function die { echo $1: status $2 ; exit $2; }
for test in $(ls $SCRAM_TEST_PATH | grep createTrackerAlignmentRcds); do
    echo -e "\n\nrunning unit test: " ${test}
    cmsRun $SCRAM_TEST_PATH/${test}  || die "Failure running ${test}" $?
done	    
