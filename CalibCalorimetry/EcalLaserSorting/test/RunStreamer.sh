#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

if [ -z  $SCRAM_TEST_PATH ]; then
SCRAM_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "LOCAL_TEST_DIR = $SCRAM_TEST_PATH"

RC=0

mkdir inDir
cmsRun ${SCRAM_TEST_PATH}/streamOutPadding_cfg.py > outp 2>&1 || die "cmsRun streamOutPadding_cfg.py" $?
cp teststreamfile.dat teststreamfile.padding
mv teststreamfile.dat inDir/
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamIn_cfg.py  > inp  2>&1 || die "cmsRun streamIn_cfg.py" $?
rm -rf inDir

mkdir inDir
cmsRun ${SCRAM_TEST_PATH}/streamOut_cfg.py > out 2>&1 || die "cmsRun streamOut_cfg.py" $?
cp teststreamfile.dat teststreamfile.original
mv teststreamfile.dat inDir
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamIn_cfg.py  > in  2>&1 || die "cmsRun streamIn_cfg.py" $?

rm watcherSourceToken
cp teststreamfile.original inDir/teststreamfile.dat
cmsRun ${SCRAM_TEST_PATH}/streamOutAlt_cfg.py  > outAlt 2>&1 || die "cmsRun streamOutAlt_cfg.py" $?
mv teststreamfile_alt.dat inDir
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamIn_cfg.py  >alt  2>&1 || die "cmsRun streamIn_cfg.py" $?
#timeout --signal SIGTERM 180 cmsRun  ${SCRAM_TEST_PATH}/streamInAlt_cfg.py  > alt  2>&1 || die "cmsRun streamInAlt_cfg.py" $?

rm watcherSourceToken
cp teststreamfile.original inDir/teststreamfile.dat
cmsRun ${SCRAM_TEST_PATH}/streamOutExt_cfg.py  > outExt 2>&1 || die "cmsRun streamOutExt_cfg.py" $?
mv teststreamfile_ext.dat inDir
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamIn_cfg.py  > ext  2>&1 || die "cmsRun streamIn_cfg.py" $?
#timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamInExt_cfg.py  > ext  2>&1 || die "cmsRun streamInExt_cfg.py" $?

# echo "CHECKSUM = 1" > out
# echo "CHECKSUM = 1" > in

ANS_OUT_SIZE=`grep -c CHECKSUM out`
ANS_OUT=`grep CHECKSUM out`
ANS_IN=`grep CHECKSUM in`

ANS_OUTP_SIZE=`grep -c CHECKSUM outp`
ANS_OUTP=`grep CHECKSUM outp`
ANS_INP=`grep CHECKSUM inp`

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

if [ "${ANS_OUTP_SIZE}" == "0" ]
then
    echo "New Stream Test Failed (out was not created)"
    RC=1
fi

if [ "${ANS_OUTP}" != "${ANS_INP}" ]
then
    echo "New Stream Test Failed (out!=in)"
    RC=1
fi

exit ${RC}
