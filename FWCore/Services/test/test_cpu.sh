#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${SCRAM_TEST_PATH}/test_CPU.py

(cmsRun -j test_CPU.xml $F1 ) || die "Failure using $F1" $?
grep -v '""' test_CPU.xml | grep -q CPUModels || die "CPUModels not found using $F1" $?
