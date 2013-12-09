#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_deepCall_allowUnscheduled_true_cfg.py
F2=${LOCAL_TEST_DIR}/test_deepCall_allowUnscheduled_true_fail_cfg.py
F3=${LOCAL_TEST_DIR}/test_offPath_allowUnscheduled_false_fail_cfg.py
F4=${LOCAL_TEST_DIR}/test_offPath_allowUnscheduled_true_cfg.py
F5=${LOCAL_TEST_DIR}/test_onPath_allowUnscheduled_false_cfg.py
F6=${LOCAL_TEST_DIR}/test_onPath_allowUnscheduled_true_cfg.py
F7=${LOCAL_TEST_DIR}/test_onPath_wrongOrder_allowUnscheduled_false_fail_cfg.py
F8=${LOCAL_TEST_DIR}/test_onPath_wrongOrder_allowUnscheduled_true_fail_cfg.py

(cmsRun $F1 ) > test_deepCall_allowUnscheduled_true.log || die "Failure using $F1" $?
diff ${LOCAL_TEST_DIR}/unit_test_outputs/test_deepCall_allowUnscheduled_true.log test_deepCall_allowUnscheduled_true.log || die "comparing test_deepCall_allowUnscheduled_true.log" $?

!(cmsRun $F2 ) || die "Failure using $F2" $?
!(cmsRun $F3 ) || die "Failure using $F3" $?
(cmsRun $F4 ) || die "Failure using $F4" $?
(cmsRun $F5 ) || die "Failure using $F5" $?
(cmsRun $F6 ) || die "Failure using $F6" $?
!(cmsRun $F7 ) || die "Failure using $F7" $?
!(cmsRun $F8 ) || die "Failure using $F8" $?
