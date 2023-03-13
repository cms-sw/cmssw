#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

if [ -z $SCRAM_TEST_PATH ]; then
SCRAM_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "LOCAL_TEST_DIR = $SCRAM_TEST_PATH"

RC=0

mkdir run000001
#the initial json file to read
echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_test.jsn
cmsRun ${SCRAM_TEST_PATH}/streamOut_cfg.py > out 2>&1 || die "cmsRun streamOut_cfg.py" $?
mv teststreamfile.dat run000001/teststreamfile.dat
cmsRun ${SCRAM_TEST_PATH}/streamOutAlt_cfg.py  > outAlt 2>&1 || die "cmsRun streamOutAlt_cfg.py" $?
cmsRun ${SCRAM_TEST_PATH}/streamOutExt_cfg.py  > outExt 2>&1 || die "cmsRun streamOutExt_cfg.py" $?
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamIn_cfg.py  > in  2>&1 || die "cmsRun streamIn_cfg.py" $?

echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_testAlt.jsn
mv teststreamfile_alt.dat run000001/teststreamfile_alt.dat
rm run000001/run000001_ls0000_EoR.jsn
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamInAlt_cfg.py  > alt  2>&1 || die "cmsRun streamInAlt_cfg.py" $?

echo "{\"data\" :[10,10, \"teststreamfile.dat\"]}" >  run000001/run1_ls1_testExt.jsn
rm run000001/run000001_ls0000_EoR.jsn
mv teststreamfile_ext.dat run000001/teststreamfile_ext.dat
timeout --signal SIGTERM 180 cmsRun ${SCRAM_TEST_PATH}/streamInExt_cfg.py  > ext  2>&1 || die "cmsRun streamInExt_cfg.py" $?

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

exit ${RC}
