#!/bin/sh
# Pass in name and status
function die { echo $1: status $2 ;  exit $2; }

LOCAL_TEST_DIR=$SCRAM_TEST_PATH

cmsRun ${LOCAL_TEST_DIR}/PoolOutputTest_cfg.py || die 'Failure using PoolOutputTest_cfg.py 1' $?
GUID1=$(edmFileUtil -u PoolOutputTest.root | fgrep uuid | awk '{print $10}')
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTest_cfg.py || die 'Failure using PoolOutputTest_cfg.py 2' $?
GUID2=$(edmFileUtil -u PoolOutputTest.root | fgrep uuid | awk '{print $10}')
if [ "x${GUID1}" == "x${GUID2}" ]; then
    echo "GUID from two executions are the same: ${GUID1}"
    exit 1
fi

cmsRun ${LOCAL_TEST_DIR}/PoolDropTest_cfg.py || die 'Failure using PoolDropTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolMissingTest_cfg.py || die 'Failure using PoolMissingTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolOutputRead_cfg.py || die 'Failure using PoolOutputRead_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolDropRead_cfg.py || die 'Failure using PoolDropRead_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolMissingRead_cfg.py || die 'Failure using PoolMissingRead_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolTransientTest_cfg.py || die 'Failure using PoolTransientTest_cfg.py' $?

cmsRun ${LOCAL_TEST_DIR}/PoolTransientRead_cfg.py || die 'Failure using PoolTransientRead_cfg.py' $?

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

cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef01-2345-6789-abcd-ef0123456789 || die 'Failure using PoolOutputTestOverrideGUID_cfg.py with valid GUID' $?
GUID=$(edmFileUtil -u PoolOutputTestOverrideGUID.root | fgrep uuid | awk '{print $10}')
if [ "x${GUID}" != "xabcdef01-2345-6789-abcd-ef0123456789" ]; then
    echo "GUID in file '${GUID}' did not match 'abcdef01-2345-6789-abcd-ef0123456789'"
    exit 1
fi
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid ABCDEF01-2345-6789-abcd-ef0123456789 || die 'Failure using PoolOutputTestOverrideGUID_cfg.py with valid GUID (with some capital letteters)' $?
GUID=$(edmFileUtil -u PoolOutputTestOverrideGUID.root | fgrep uuid | awk '{print $10}')
if [ "x${GUID}" != "xABCDEF01-2345-6789-abcd-ef0123456789" ]; then
    echo "GUID in file '${GUID}' did not match 'ABCDEF01-2345-6789-abcd-ef0123456789'"
    exit 1
fi

cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef01-2345-6789-abcd-ef01234567890 && die 'PoolOutputTestOverrideGUID_cfg.py with invalid GUID 1 did not fail' 1
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef01-2345-6789-0abcd-ef0123456789 && die 'PoolOutputTestOverrideGUID_cfg.py with invalid GUID 2 did not fail' 1
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid 0abcdef01-2345-6789-abcd-ef0123456789 && die 'PoolOutputTestOverrideGUID_cfg.py with invalid GUID 3 did not fail' 1
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef012-345-6789-abcd-ef0123456789 && die 'PoolOutputTestOverrideGUID_cfg.py with invalid GUID 4 did not fail' 1
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef01-2345-6789-abcd-ef012345678g && die 'PoolOutputTestOverrideGUID_cfg.py with invalid GUID 5 did not fail' 1
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef01-2345-6789-abcd_ef0123456789 && die 'PoolOutputTestOverrideGUID_cfg.py with invalid GUID 6 did not fail' 1

cmsRun ${LOCAL_TEST_DIR}/PoolOutputTest_cfg.py --firstLumi 1
cmsRun ${LOCAL_TEST_DIR}/PoolOutputTest_cfg.py --firstLumi 2

cmsRun ${LOCAL_TEST_DIR}/PoolOutputTestOverrideGUID_cfg.py --guid abcdef01-2345-6789-abcd-ef0123456789 --input PoolOutputTestLumi1.root PoolOutputTestLumi2.root --maxSize 1 || die 'Failure using PoolOutputTestOverrideGUID_cfg.py with valid GUID and two input files' $?
GUID1=$(edmFileUtil -u PoolOutputTestOverrideGUID.root | fgrep uuid | awk '{print $10}')
GUID2=$(edmFileUtil -u PoolOutputTestOverrideGUID001.root | fgrep uuid | awk '{print $10}')
if [ "x${GUID1}" != "xabcdef01-2345-6789-abcd-ef0123456789" ]; then
    echo "GUID in first file '${GUID1}' did not match 'abcdef01-2345-6789-abcd-ef0123456789'"
    exit 1
fi
if [ "x${GUID1}" == "x${GUID2}" ]; then
    echo "GUID from two output files are the same: ${GUID1}"
    exit 1
fi
