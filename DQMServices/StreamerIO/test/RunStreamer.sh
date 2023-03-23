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

cd $LOCAL_TEST_DIR

RC=0
P=$$
PREFIX=results_${USER}${P}
OUTDIR=${LOCAL_TMP_DIR}/${PREFIX}

mkdir ${OUTDIR}
cp *_cfg.py ${OUTDIR}
cd ${OUTDIR}

mkdir run000001
#the initial json file to read
echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_test.jsn
cmsRun streamOut_cfg.py > out 2>&1 || die "cmsRun streamOut_cfg.py" $?
mv teststreamfile.dat run000001/teststreamfile.dat
cmsRun streamOutAlt_cfg.py  > outAlt 2>&1 || die "cmsRun streamOutAlt_cfg.py" $?
cmsRun streamOutExt_cfg.py  > outExt 2>&1 || die "cmsRun streamOutExt_cfg.py" $?
timeout --signal SIGTERM 180 cmsRun streamIn_cfg.py  > in  2>&1 || die "cmsRun streamIn_cfg.py" $?

echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_testAlt.jsn
mv teststreamfile_alt.dat run000001/teststreamfile_alt.dat
rm run000001/run000001_ls0000_EoR.jsn
timeout --signal SIGTERM 180 cmsRun  streamInAlt_cfg.py  > alt  2>&1 || die "cmsRun streamInAlt_cfg.py" $?

echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_testExt.jsn
rm run000001/run000001_ls0000_EoR.jsn
mv teststreamfile_ext.dat run000001/teststreamfile_ext.dat
timeout --signal SIGTERM 180 cmsRun streamInExt_cfg.py  > ext  2>&1 || die "cmsRun streamInExt_cfg.py" $?

echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_test.jsn
cmsRun streamOutPadding_cfg.py > outPadding 2>&1 || die "cmsRun streamOutPadding_cfg.py" $?
mv teststreamfile_padding.dat run000001/teststreamfile.dat
timeout --signal SIGTERM 180 cmsRun streamIn_cfg.py  > inPadding  2>&1 || die "cmsRun streamIn_cfg.py" $?


# echo "CHECKSUM = 1" > out
# echo "CHECKSUM = 1" > in

ANS_OUT_SIZE=`grep -c CHECKSUM out`
ANS_OUT_PADD_SIZE=`grep -c CHECKSUM outPadding`
ANS_OUT=`grep CHECKSUM out`
ANS_OUT_PADD=`grep CHECKSUM outPadding`
ANS_IN=`grep CHECKSUM in`
ANS_IN_PADD=`grep CHECKSUM inPadding`

if [ "${ANS_OUT_SIZE}" == "0" ]
then
    echo "New Stream Test Failed (out was not created)"
    RC=1
fi

if [ "${ANS_OUT_PADD_SIZE}" == "0" ]
then
    echo "New Padded Stream Test Failed (out was not created)"
    RC=1
fi

if [ "${ANS_OUT}" != "${ANS_IN}" ]
then
    echo "New Stream Test Failed (out!=in)"
    RC=1
fi

if [ "${ANS_OUT_PADD}" != "${ANS_IN_PADD}" ]
then
    echo "New Padded Stream Test Failed (out!=in)"
    RC=1
fi

exit ${RC}
