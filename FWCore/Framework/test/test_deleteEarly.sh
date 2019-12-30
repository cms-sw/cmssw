#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/test_doNotDeleteEarly_cfg.py
F2=${LOCAL_TEST_DIR}/test_simpleDeleteEarly_cfg.py
F3=${LOCAL_TEST_DIR}/test_readAfterEarlyDelete_fail_cfg.py
F4=${LOCAL_TEST_DIR}/test_multiPathEarlyDelete_cfg.py
F5=${LOCAL_TEST_DIR}/test_multiPathMultiModuleEarlyDelete_cfg.py
F6=${LOCAL_TEST_DIR}/test_subProcessDeleteEarly_cfg.py
F7=${LOCAL_TEST_DIR}/test_consumeAfterEarlyDeleteTask_cfg.py
F8=${LOCAL_TEST_DIR}/test_consumeAfterEarlyDeletePath_cfg.py

(cmsRun $F1 ) || die "Failure using $F1" $?
(cmsRun $F2 ) || die "Failure using $F2" $?
!(cmsRun $F3 ) || die "Failure using $F3" $?
(cmsRun $F4 ) || die "Failure using $F4" $?
(cmsRun $F5 ) || die "Failure using $F5" $?
(cmsRun $F6 ) || die "Failure using $F6" $?
(cmsRun $F7 ) || die "Failure using $F7" $?
(cmsRun $F8 ) || die "Failure using $F8" $?


