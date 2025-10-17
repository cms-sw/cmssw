#!/bin/sh

function die { echo $1: status $2 ;  exit $2; }
function diecat { echo "$1: status $2, log" ;  cat $3; exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

echo testConcurrentIOVsAndRuns_cfg.py
cmsRun ${LOCAL_TEST_DIR}/testConcurrentIOVsAndRuns_cfg.py || die 'Failed in testConcurrentIOVsAndRuns_cfg.py' $?

echo testConcurrentIOVsAndRunsRead_cfg.py
cmsRun ${LOCAL_TEST_DIR}/testConcurrentIOVsAndRunsRead_cfg.py || die 'Failed in testConcurrentIOVsAndRunsRead_cfg.py' $?

