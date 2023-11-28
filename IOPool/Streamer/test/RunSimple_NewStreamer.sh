#!/bin/bash

function die { echo Failure $1: status $2 ; echo ""; cat log ; exit $2 ; }

if [ -z  $SCRAM_TEST_PATH ]; then
SCRAM_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "LOCAL_TEST_DIR = $SCRAM_TEST_PATH"

if [ -z  $TEST_COMPRESSION_ALGO ]; then
TEST_COMPRESSION_ALGO="ZLIB"
fi
echo "TEST_COMPRESSION_ALGO = $TEST_COMPRESSION_ALGO"

RC=0

rm -rf {out,outPadded,log,*.txt,*.dat,*.root}

<<<<<<< HEAD
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
=======
cmsRun ${SCRAM_TEST_PATH}/NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutAlt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutExt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutExt2_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOutExt2_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamOutPadding_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > log 2>&1 || die "cmsRun NewStreamOutPadding_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamIn_cfg.py  > log  2>&1 || die "cmsRun NewStreamIn_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamIn2_cfg.py  > log  2>&1 || die "cmsRun NewStreamIn2_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamCopy_cfg.py  > log  2>&1 || die "cmsRun NewStreamCopy_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamCopy2_cfg.py  > log  2>&1 || die "cmsRun NewStreamCopy2_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInAlt_cfg.py  > log  2>&1 || die "cmsRun NewStreamInAlt_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInExt_cfg.py  > log  2>&1 || die "cmsRun NewStreamInExt_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInExtBuf_cfg.py > log  2>&1 || die "cmsRun NewStreamInExtBuf_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInPadding_cfg.py > log  2>&1 || die "cmsRun NewStreamInPadding_cfg.py (1)" $?
cmsRun ${SCRAM_TEST_PATH}/NewStreamInPadding_cfg.py inChecksum=outPadded  > log  2>&1 || die "cmsRun NewStreamInPadding_cfg.py (2)" $?
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704

# echo "CHECKSUM = 1" > out

if [ ! -s out ]; then

    echo "New Stream Test Failed (out was not created or is empty)"
    RC=1
fi

<<<<<<< HEAD
#rm -rf ${OUTDIR}
=======
>>>>>>> 895df58e36cff1d7dc27b1bf37aee7f604adc704
exit ${RC}
