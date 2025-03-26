#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH

cmsRun ${LOCAL_TEST_DIR}/PoolOutputAliasTest_cfg.py || die 'Failure using PoolOutputAliasTest_cfg.py 1' $?
python3 ${LOCAL_TEST_DIR}/PoolOutputAliasTestCheckResults.py || die 'results check failed' $?