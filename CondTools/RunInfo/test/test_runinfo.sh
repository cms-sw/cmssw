#!/bin/sh

LOCAL_TEST_DIR=$CMSSW_BASE/src/CondTools/RunInfo/test

function die { echo Failure $1: status $2 ; exit $2 ; }

cmsRun ${LOCAL_TEST_DIR}/test_runinfo_cfg.py | grep '\(run number\)\|\(average current\)' > test_runinfo.run_log || die "cmsRun RefTest_cfg.py" $?
diff test_runinfo.run_log ${LOCAL_TEST_DIR}/test_runinfo_result.log || die 'incorrect output using test_runinfo_cfg.py' $?
rm test_runinfo.run_log
