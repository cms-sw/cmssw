#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${SCRAM_TEST_PATH}/HcalPulseContainmentTest_cfg.py
(cmsRun $F1 ) || die "Failure using $F1" $?