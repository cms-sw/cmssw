#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

echo "LOCAL_TEST_DIR = $LOCAL_TEST_DIR"
echo "LOCAL_TMP_DIR = $LOCAL_TMP_DIR"

cd $LOCAL_TEST_DIR

RC=0
P=$$
PREFIX=results_${USER}${P}
OUTDIR=${LOCAL_TMP_DIR}/${PREFIX}

mkdir ${OUTDIR}
cp *_cfg.py ${OUTDIR}
cd ${OUTDIR}

cmsRun --parameter-set NewStreamOut_cfg.py > out 2>&1 || die "cmsRun NewStreamOut_cfg.py" $?
cmsRun --parameter-set NewStreamIn_cfg.py  > in  2>&1 || die "cmsRun NewStreamIn_cfg.py" $?
cmsRun --parameter-set NewStreamIn2_cfg.py  > in2  2>&1 || die "cmsRun NewStreamIn2_cfg.py" $?
cmsRun --parameter-set NewStreamCopy_cfg.py  > copy  2>&1 || die "cmsRun NewStreamCopy_cfg.py" $?
cmsRun --parameter-set NewStreamCopy2_cfg.py  > copy2  2>&1 || die "cmsRun NewStreamCopy2_cfg.py" $?

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
