#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/testTrigBits0_cfg.py  
F2=${LOCAL_TEST_DIR}/testTrigBits1_cfg.py  
F3=${LOCAL_TEST_DIR}/testTrigBits2_cfg.py  
F4=${LOCAL_TEST_DIR}/testTrigBits3_cfg.py  
F5=${LOCAL_TEST_DIR}/testTrigBits4_cfg.py

(cmsRun $F1 ) || die "Failure using $F1" $?
(cmsRun $F2 ) || die "Failure using $F2" $?
(cmsRun $F3 ) || die "Failure using $F3" $?
(cmsRun $F4 ) || die "Failure using $F4" $?
(cmsRun $F5 ) || die "Failure using $F5" $?


