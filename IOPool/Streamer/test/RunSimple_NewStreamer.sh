#!/bin/bash

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

cmsRun --parameter-set NewStreamOut_cfg.py > out 2>&1
cmsRun --parameter-set NewStreamIn_cfg.py  > in  2>&1
cmsRun --parameter-set NewStreamCopy_cfg.py  > copy  2>&1

# echo "CHECKSUM = 1" > out
# echo "CHECKSUM = 1" > in

ANS_OUT_SIZE=`grep -c CHECKSUM out`
ANS_OUT=`grep CHECKSUM out`
ANS_IN=`grep CHECKSUM in`
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

if [ "${ANS_OUT}" != "${ANS_COPY}" ]
then
    echo "New Stream Test Failed (copy!=out)"
    RC=1
fi

#rm -rf ${OUTDIR}
exit ${RC}
