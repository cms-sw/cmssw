#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

F1=${LOCAL_TEST_DIR}/test_deepCall_unscheduled_cfg.py
F2=${LOCAL_TEST_DIR}/test_deepCall_unscheduled_fail_cfg.py
F3=${LOCAL_TEST_DIR}/test_offPath_unscheduled_cfg.py
F4=${LOCAL_TEST_DIR}/test_onPath_unscheduled_cfg.py
F5=${LOCAL_TEST_DIR}/test_onPath_wrongOrder_unscheduled_fail_cfg.py

(cmsRun $F1 ) > test_deepCall_unscheduled.log || die "Failure using $F1" $?
diff ${LOCAL_TEST_DIR}/unit_test_outputs/test_deepCall_unscheduled.log test_deepCall_unscheduled.log || die "comparing test_deepCall_unscheduled.log" $?

!(cmsRun $F2 ) || die "Failure using $F2" $?
(cmsRun $F3 ) || die "Failure using $F3" $?

(cmsRun $F4 )  > test_onPath_unscheduled.log || die "Failure using $F4" $?
diff ${LOCAL_TEST_DIR}/unit_test_outputs/test_onPath_unscheduled.log test_onPath_unscheduled.log || die "comparing test_onPath_unscheduled.log" $?

!(cmsRun $F5 ) || die "Failure using $F5" $?

popd

