#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function diewrite {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_write.log;  rm -rf $3/{ramdisk,data,*.py}; exit $2 ; }
function dieread {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_read.log;  rm -rf $3/{ramdisk,data,*.py}; exit $2 ; }
function diemerge {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_merge.log;  rm -rf $3/{ramdisk,data,*.py}; exit $2 ; }



if [ -z  ${SCRAM_TEST_PATH} ]; then
SCRAM_TEST_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "SCRAM_TEST_PATH = ${SCRAM_TEST_PATH}"

RC=0
P=$$
PREFIX=results_${USER}${P}
OUTDIR=${PWD}/${PREFIX}

echo "OUT_TMP_DIR = $OUTDIR"

mkdir ${OUTDIR}
cp ${SCRIPTDIR}/writeStreamerFile_cfg.py ${OUTDIR}
cp ${SCRIPTDIR}/readStreamerFile_cfg.py ${OUTDIR}
cp ${SCRIPTDIR}/mergeStreamerFile_cfg.py ${OUTDIR}
cd ${OUTDIR}

rm -rf $OUTDIR/{ramdisk,data,*.log}

runnumber="100101"

###############################
echo "Running test on reading single file"
CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat --runNumber=${runnumber} --numEvents=10"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR

rm -rf data
##########################
echo "Running test on reading two separate files"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test1.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test1.dat

rm -rf data

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test2.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test2.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test1.dat --input test2.dat --runNumber=${runnumber} --numEvents=20"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR

rm -rf data
#############################
echo "Running test on reading two separate files with different BranchIDLists"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test1.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test1.dat

rm -rf data

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber} --changeBranchIDLists T"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test2.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test2.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test1.dat --input test2.dat --runNumber=${runnumber} --numEvents=20"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR

rm -rf data
##########################

echo "Running test one concatenated file"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat --runNumber=${runnumber} --numEvents=20"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR
#cat out_2_read.log

rm -rf data
#############################
echo "Running test on concatenated file from jobs with different BranchIDLists"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber} --changeBranchIDLists T"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat --runNumber=${runnumber} --numEvents=20"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR
#cat out_2_read.log

rm -rf data

###########################
echo "Test merging streamers using cmsRun"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test1.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test1.dat

rm -rf data

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test2.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test2.dat

rm -rf data


CMDLINE_MERGE="cmsRun mergeStreamerFile_cfg.py --input test1.dat --input test2.dat --runNumber=${runnumber}"
${CMDLINE_MERGE} > out_2_merge.log 2>&1 || diemerge "${CMDLINE_MERGE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_merge_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_merge_pid*.dat >> test.dat


CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat  --runNumber=${runnumber} --numEvents=20"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR

rm -rf data
#############################
echo "Test merging files with different BranchIDLists"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test1.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test1.dat

rm -rf data

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber} --changeBranchIDLists T"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test2.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test2.dat

rm -rf data

CMDLINE_MERGE="cmsRun mergeStreamerFile_cfg.py --input test1.dat --input test2.dat --runNumber=${runnumber}"
${CMDLINE_MERGE} > out_2_merge.log 2>&1 || diemerge "${CMDLINE_MERGE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_merge_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_merge_pid*.dat >> test.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat --runNumber=${runnumber} --numEvents=20"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR


###############################
echo "Running test on reading single empty file"
CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=0 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat --runNumber=${runnumber} --numEvents=0"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR

rm -rf data
##########################
echo "Running test on reading two separate empty files"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --runNumber=${runnumber} --numEvents=0"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test1.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test1.dat

rm -rf data

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=10 --startEvent=11 --runNumber=${runnumber} --numEvents=0"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test2.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test2.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test1.dat --input test2.dat --runNumber=${runnumber} --numEvents=0"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR

rm -rf data
##########################

echo "Running test one concatenated empty file"

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=0 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

CMDLINE_WRITE="cmsRun writeStreamerFile_cfg.py --numEvents=0 --startEvent=11 --runNumber=${runnumber}"
${CMDLINE_WRITE}  > out_2_write.log 2>&1 || diewrite "${CMDLINE_WRITE}" $? $OUTDIR

#prepare file to read
ls -1 data/run${runnumber}/run${runnumber}_ls0000_streamA_pid*.ini | head -1 | xargs cat > test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamA_pid*.dat >> test.dat

CMDLINE_READ="cmsRun readStreamerFile_cfg.py --input test.dat --runNumber=${runnumber} --numEvents=0"
${CMDLINE_READ} > out_2_read.log 2>&1 || dieread "${CMDLINE_READ}" $? $OUTDIR
#cat out_2_read.log

rm -rf data
############################

#no failures, clean up everything including logs if there are no errors
echo "Completed sucessfully"
#rm -rf $OUTDIR/{ramdisk,data,*.py,*.log}
rm -rf $OUTDIR

exit ${RC}

