#!/bin/bash

# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

F1=${LOCAL_TEST_DIR}/testTrigBits0.cfg  
F2=${LOCAL_TEST_DIR}/testTrigBits1.cfg  
F3=${LOCAL_TEST_DIR}/testTrigBits2.cfg  
F4=${LOCAL_TEST_DIR}/testTrigBits3.cfg  
F5=${LOCAL_TEST_DIR}/testTrigBits4.cfg

(cmsRun $F1 ) || die 'Failure using $F1' $?
(cmsRun $F2 ) || die 'Failure using $F2' $?
(cmsRun $F3 ) || die 'Failure using $F3' $?
(cmsRun $F4 ) || die 'Failure using $F4' $?
(cmsRun $F5 ) || die 'Failure using $F5' $?


