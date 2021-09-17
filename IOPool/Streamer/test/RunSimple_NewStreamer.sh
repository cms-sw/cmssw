#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

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

cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > out 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamOutAlt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > outAlt 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun NewStreamOutExt_cfg.py compAlgo=${TEST_COMPRESSION_ALGO} > outExt 2>&1 || die "cmsRun NewStreamOut_cfg.py compAlgo=${TEST_COMPRESSION_ALGO}" $?
cmsRun --parameter-set NewStreamIn_cfg.py  > in  2>&1 || die "cmsRun NewStreamIn_cfg.py" $?
cmsRun --parameter-set NewStreamIn2_cfg.py  > in2  2>&1 || die "cmsRun NewStreamIn2_cfg.py" $?
cmsRun --parameter-set NewStreamCopy_cfg.py  > copy  2>&1 || die "cmsRun NewStreamCopy_cfg.py" $?
cmsRun --parameter-set NewStreamCopy2_cfg.py  > copy2  2>&1 || die "cmsRun NewStreamCopy2_cfg.py" $?
cmsRun --parameter-set NewStreamInAlt_cfg.py  > alt  2>&1 || die "cmsRun NewStreamInAlt_cfg.py" $?
cmsRun --parameter-set NewStreamInExt_cfg.py  > ext  2>&1 || die "cmsRun NewStreamInExt_cfg.py" $?

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

#rm -rf ${OUTDIR}
exit ${RC}
