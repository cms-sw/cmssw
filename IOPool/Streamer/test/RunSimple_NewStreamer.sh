#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

if [ -z  $SCRAM_TEST_PATH ]; then
SCRAM_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "LOCAL_TEST_DIR = $SCRAM_TEST_PATH"

if [ -z  $TEST_COMPRESSION_ALGO ]; then
TEST_COMPRESSION_ALGO="ZLIB"
fi
echo "TEST_COMPRESSION_ALGO = $TEST_COMPRESSION_ALGO"

RC=0

cmsRun ${SCRAM_TEST_PATH}/NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > out 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutAlt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > outAlt 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutExt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > outExt 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutExt2_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > outExt 2>&1 || die "cmsRun NewStreamOutExt2_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamIn_cfg.py  > in  2>&1 || die "cmsRun NewStreamIn_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamIn2_cfg.py  > in2  2>&1 || die "cmsRun NewStreamIn2_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamCopy_cfg.py  > copy  2>&1 || die "cmsRun NewStreamCopy_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamCopy2_cfg.py  > copy2  2>&1 || die "cmsRun NewStreamCopy2_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInAlt_cfg.py  > alt  2>&1 || die "cmsRun NewStreamInAlt_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInExt_cfg.py  > ext  2>&1 || die "cmsRun NewStreamInExt_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInExtBuf_cfg.py  > ext  2>&1 || die "cmsRun NewStreamInExtBuf_cfg.py" $?

# echo "CHECKSUM = 1" > out
# echo "CHECKSUM = 1" > in

ANS_OUT_SIZE=`grep -c CHECKSUM out`
ANS_OUT=`grep CHECKSUM out`
ANS_IN=`grep CHECKSUM in`
ANS_IN2=`grep CHECKSUM in2`
ANS_COPY=`grep CHECKSUM copy`

if [ "${ANS_OUT_SIZE}" == "0" ]
then
    echo "New Stream Test Failed (out was not created)"
    RC=1
fi

if [ "${ANS_OUT}" != "${ANS_IN}" ]
then
    echo "New Stream Test Failed (out!=in)"
    RC=1
fi

if [ "${ANS_OUT}" != "${ANS_IN2}" ]
then
    echo "New Stream Test Failed (out!=in2)"
    RC=1
fi

if [ "${ANS_OUT}" != "${ANS_COPY}" ]
then
    echo "New Stream Test Failed (copy!=out)"
    RC=1
fi

exit ${RC}
