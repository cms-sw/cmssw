#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolOutputTest_cfg.py || die 'Failure using PoolOutputTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolDropTest_cfg.py || die 'Failure using PoolDropTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolMissingTest_cfg.py || die 'Failure using PoolMissingTest_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolOutputRead_cfg.py || die 'Failure using PoolOutputRead_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolDropRead_cfg.py || die 'Failure using PoolDropRead_cfg.py' $?

cmsRun --parameter-set ${LOCAL_TEST_DIR}/PoolMissingRead_cfg.py || die 'Failure using PoolMissingRead_cfg.py' $?


cmsRun ${LOCAL_TEST_DIR}/TestProvA_cfg.py || die 'Failure using TestProvA_cfg.py' $?
#reads file from above
cmsRun ${LOCAL_TEST_DIR}/TestProvB_cfg.py || die 'Failure using TestProvB_cfg.py' $?
#reads file from above
cmsRun ${LOCAL_TEST_DIR}/TestProvC_cfg.py || die 'Failure using TestProvC_cfg.py' $?
