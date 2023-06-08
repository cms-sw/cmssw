#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_Run2Scouting_test_file_cfg.py || die 'Failure using create_Run2Scouting_test_file_cfg.py' $?

file=testRun2Scouting.root

cmsRun ${LOCAL_TEST_DIR}/test_readRun2Scouting_cfg.py "$file" || die "Failure using test_readRun2Scouting_cfg.py $file" $?

exit 0
