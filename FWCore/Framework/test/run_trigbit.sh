#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/testBitsPass_cfg.py
F2=${LOCAL_TEST_DIR}/testBitsFail_cfg.py
F3=${LOCAL_TEST_DIR}/testBitsMove_cfg.py
F4=${LOCAL_TEST_DIR}/testBitsCount_cfg.py
F5=${LOCAL_TEST_DIR}/testFilterIgnore_cfg.py
F6=${LOCAL_TEST_DIR}/testFilterOnEndPath_cfg.py

(cmsRun $F1 ) || die "Failure using $F1" $?
(cmsRun $F2 ) || die "Failure using $F2" $?
(cmsRun $F3 ) || die "Failure using $F3" $?
(cmsRun $F4 ) || die "Failure using $F4" $?
(cmsRun $F5 ) || die "Failure using $F5" $?
(cmsRun $F6 ) || die "Failure using $F6" $?


