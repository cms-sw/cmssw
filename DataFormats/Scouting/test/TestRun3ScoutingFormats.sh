#!/bin/sh -ex

function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun ${LOCAL_TEST_DIR}/create_Run3Scouting_test_file.py || die 'Failure using create_Run3Scouting_test_file.py' $?

file=testRun3Scouting.root

cmsRun ${LOCAL_TEST_DIR}/test_readRun3Scouting_cfg.py "$file" || die "Failure using test_readRun3Scouting_cfg.py $file" $?

exit 0
