#!/bin/bash

function die { echo Failure $1: status $2 ; echo ""; cat log ; exit $2 ; }

if [ -z  $LOCAL_TEST_DIR ]; then
LOCAL_TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "LOCAL_TEST_DIR = $LOCAL_TEST_DIR"

if [ -z  $LOCAL_TMP_DIR ]; then
LOCAL_TMP_DIR="/tmp"
fi
echo "LOCAL_TMP_DIR = $LOCAL_TMP_DIR"

if [ -z  $TEST_COMPRESSION_ALGO ]; then
TEST_COMPRESSION_ALGO="ZLIB"
fi
echo "TEST_COMPRESSION_ALGO = $TEST_COMPRESSION_ALGO"

cd $LOCAL_TEST_DIR

RC=0
P=$$
PREFIX=results_${USER}${P}
OUTDIR=${LOCAL_TMP_DIR}/${PREFIX}

mkdir ${OUTDIR}
cp *_cfg.py ${OUTDIR}
cd ${OUTDIR}

cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamOutAlt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamOutExt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamOutExt2_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOutExt2_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamOutPadding_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOutPadding_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamIn_cfg.py  > log  2>&1 || die "cmsRun NewStreamIn_cfg.py" $?
cmsRun NewStreamIn2_cfg.py  > log  2>&1 || die "cmsRun NewStreamIn2_cfg.py" $?
cmsRun NewStreamCopy_cfg.py  > log  2>&1 || die "cmsRun NewStreamCopy_cfg.py" $?
cmsRun NewStreamCopy2_cfg.py  > log  2>&1 || die "cmsRun NewStreamCopy2_cfg.py" $?
cmsRun NewStreamInAlt_cfg.py  > log  2>&1 || die "cmsRun NewStreamInAlt_cfg.py" $?
cmsRun NewStreamInExt_cfg.py  > log  2>&1 || die "cmsRun NewStreamInExt_cfg.py" $?
cmsRun NewStreamInExtBuf_cfg.py > log  2>&1 || die "cmsRun NewStreamInExtBuf_cfg.py" $?
cmsRun NewStreamInPadding_cfg.py > log  2>&1 || die "cmsRun NewStreamInPadding_cfg.py (1)" $?
cmsRun NewStreamInPadding_cfg.py inChecksum=outPadded  > log  2>&1 || die "cmsRun NewStreamInPadding_cfg.py (2)" $?

# echo "CHECKSUM = 1" > out

if [ ! -s out ]; then

    echo "New Stream Test Failed (out was not created or is empty)"
    RC=1
fi

#rm -rf ${OUTDIR}
exit ${RC}
