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

mkdir inDir
cmsRun streamOut_cfg.py > out 2>&1 || die "cmsRun streamOut_cfg.py" $?
cp teststreamfile.dat teststreamfile.original
mv teststreamfile.dat inDir
timeout --signal SIGTERM 180 cmsRun streamIn_cfg.py  > in  2>&1 || die "cmsRun streamIn_cfg.py" $?

rm watcherSourceToken
cp teststreamfile.original inDir/teststreamfile.dat
cmsRun streamOutAlt_cfg.py  > outAlt 2>&1 || die "cmsRun streamOutAlt_cfg.py" $?
mv teststreamfile_alt.dat inDir
timeout --signal SIGTERM 180 cmsRun streamIn_cfg.py  >alt  2>&1 || die "cmsRun streamIn_cfg.py" $?
#timeout --signal SIGTERM 180 cmsRun  streamInAlt_cfg.py  > alt  2>&1 || die "cmsRun streamInAlt_cfg.py" $?

rm watcherSourceToken
cp teststreamfile.original inDir/teststreamfile.dat
cmsRun streamOutExt_cfg.py  > outExt 2>&1 || die "cmsRun streamOutExt_cfg.py" $?
mv teststreamfile_ext.dat inDir
timeout --signal SIGTERM 180 cmsRun streamIn_cfg.py  > ext  2>&1 || die "cmsRun streamIn_cfg.py" $?
#timeout --signal SIGTERM 180 cmsRun streamInExt_cfg.py  > ext  2>&1 || die "cmsRun streamInExt_cfg.py" $?

# echo "CHECKSUM = 1" > out
# echo "CHECKSUM = 1" > in

ANS_OUT_SIZE=`grep -c CHECKSUM out`
ANS_OUT=`grep CHECKSUM out`
ANS_IN=`grep CHECKSUM in`

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

#rm -rf ${OUTDIR}
exit ${RC}
