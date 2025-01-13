#!/bin/sh -ex
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

cmsRun -j PoolInputTest_jobreport.xml ${LOCAL_TEST_DIR}/PrePoolInputTest_cfg.py PoolInputTest.root 11 561 7 6 3 || die 'Failure using PrePoolInputTest_cfg.py' $?

cmsRun  -j PoolGuidTest_jobreport.xml ${LOCAL_TEST_DIR}/PoolGUIDTest_cfg.py file:PoolInputTest.root && die 'PoolGUIDTest_cfg.py PoolInputTest.root did not throw an exception' 1
GUID_EXIT_CODE=$(edmFjrDump --exitCode PoolGuidTest_jobreport.xml)
if [ "x${GUID_EXIT_CODE}" != "x8034" ]; then
    echo "Inconsistent GUID test reported exit code ${GUID_EXIT_CODE} which is different from the expected 8034"
    exit 1
fi
GUID_NAME=$(edmFjrDump --guid PoolInputTest_jobreport.xml).root
cp PoolInputTest.root ${GUID_NAME}
cmsRun ${LOCAL_TEST_DIR}/PoolGUIDTest_cfg.py file:${GUID_NAME} || die 'Failure using PoolGUIDTest_cfg.py ${GUID_NAME}' $?


cp PoolInputTest.root PoolInputOther.root

cmsRun ${LOCAL_TEST_DIR}/PoolInputTest_cfg.py || die 'Failure using PoolInputTest_cfg.py' $?
cmsRun  ${LOCAL_TEST_DIR}/PoolInputTest_noDelay_cfg.py >& PoolInputTest_noDelay_cfg.txt || die 'Failure using PoolInputTest_noDelay_cfg.py' $?
grep 'event delayed read from source' PoolInputTest_noDelay_cfg.txt && die 'Failure in PoolInputTest_noDelay_cfg.py, found delay reads from source' 1
cmsRun ${LOCAL_TEST_DIR}/PoolInputTest_skip_with_failure_cfg.py || die 'Failure using PoolInputTest_skip_with_failure_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/PoolInputTest_skipBadFiles_cfg.py  || die 'Failure using PoolInputTest_skipBadFiles_cfg.py ' $?

cmsRun ${LOCAL_TEST_DIR}/PrePool2FileInputTest_cfg.py || die 'Failure using PrePool2FileInputTest_cfg.py' $?
cmsRun ${LOCAL_TEST_DIR}/Pool2FileInputTest_cfg.py || die 'Failure using Pool2FileInputTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PrePoolInputTest2_cfg.py || die 'Failure using PrePoolInputTest2_cfg.py' $?

cp PoolInputTest.root PoolInputOther.root

cmsRun ${LOCAL_TEST_DIR}/PoolInputTest2_cfg.py || die 'Failure using PoolInputTest2_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolInputTest3_cfg.py || die 'Failure using PoolInputTest3_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolEmptyTest_cfg.py || die 'Failure using PoolEmptyTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolEmptyTest2_cfg.py || die 'Failure using PoolEmptyTest2_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep1_cfg.py || die 'Failure using PoolAliasTestStep1_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep2_cfg.py || die 'Failure using PoolAliasTestStep2_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep1_DifferentOrder_cfg.py || die 'Failure using PoolAliasTestStep1_DifferentOrder_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep2_DifferentOrder_cfg.py || die 'Failure using PoolAliasTestStep2_DifferentOrder_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep2A_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep1C_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasTestStep2C_cfg.py || die 'Failure using PoolAliasTestStep2A_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasSubProcessTestStep1_cfg.py || die 'Failure using PoolAliasSubProcessTestStep1_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolAliasSubProcessTestStep2_cfg.py || die 'Failure using PoolAliasSubProcessTestStep2_cfg.py' $?

