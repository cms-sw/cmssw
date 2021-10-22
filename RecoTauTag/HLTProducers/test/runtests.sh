#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }

cmsRun ${LOCAL_TEST_DIR}/testL2TauTagNN.py maxEvents=10 || die 'Failure using testL2TauTagNN.py' $?
