#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

pushd ${LOCAL_TMP_DIR}

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolOutputTest_cfg.py || die 'Failure using PoolOutputTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolDropTest_cfg.py || die 'Failure using PoolDropTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolMissingTest_cfg.py || die 'Failure using PoolMissingTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolOutputRead_cfg.py || die 'Failure using PoolOutputRead_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolDropRead_cfg.py || die 'Failure using PoolDropRead_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolMissingRead_cfg.py || die 'Failure using PoolMissingRead_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolTransientTest_cfg.py || die 'Failure using PoolTransientTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolTransientRead_cfg.py || die 'Failure using PoolTransientRead_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolOutputEmptyEventsTest_cfg.py || die 'Failure using PoolOutputEmptyEventsTest_cfg.py' $?
#reads file from above and from PoolOutputTest_cfg.py
cmsRun ${LOCAL_TEST_DIR}/PoolOutputMergeWithEmptyFile_cfg.py || die 'Failure using PoolOutputMergeWithEmptyFile_cfg.py' $? 

cmsRun ${LOCAL_TEST_DIR}/TestProvA_cfg.py || die 'Failure using TestProvA_cfg.py' $?
#reads file from above
cmsRun ${LOCAL_TEST_DIR}/TestProvB_cfg.py || die 'Failure using TestProvB_cfg.py' $?
#reads file from above
cmsRun ${LOCAL_TEST_DIR}/TestProvC_cfg.py || die 'Failure using TestProvC_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestUnscheduled_cfg.py || die 'Failure using PoolOutputTestUnscheduled_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestUnscheduledRead_cfg.py || die 'Failure using PoolOutputTestUnscheduledRead_cfg.py' $?

popd
