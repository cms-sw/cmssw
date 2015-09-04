#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_mayConsumes_deadlocking_cfg.py
(cmsRun $F1 ) || die "Failure using $F1" $?

