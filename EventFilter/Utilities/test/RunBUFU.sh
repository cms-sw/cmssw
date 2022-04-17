
#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function diebu {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_bu.log;  rm -rf $3/{ramdisk,data,dqmdisk,*.py}; exit $2 ; }
function diefu {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_fu.log;  rm -rf $3/{ramdisk,data,dqmdisk,*.py}; exit $2 ; }
function diedqm { echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_dqm.log; rm -rf $3/{ramdisk,data,dqmdisk,*.py}; exit $2 ; }

FUSCRIPT="unittest_FU.py"
if [ ! -z $1 ]; then
  if [ "$1" == "local" ]; then
    FUSCRIPT="startFU.py"
    echo "local run: using ${FUSCRIPT}"
  fi
fi

if [ -z  $LOCAL_TEST_DIR ]; then
LOCAL_TEST_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi
echo "LOCAL_TEST_DIR = $LOCAL_TEST_DIR"

if [ -z  $LOCAL_TMP_DIR ]; then
LOCAL_TMP_DIR="/tmp"
fi

cd $LOCAL_TEST_DIR

RC=0
P=$$
PREFIX=results_${USER}${P}
OUTDIR=${LOCAL_TMP_DIR}/${PREFIX}

echo "OUT_TMP_DIR = $OUTDIR"

mkdir ${OUTDIR}
cp ${SCRIPTDIR}/startBU.py ${OUTDIR}
cp ${SCRIPTDIR}/startFU.py ${OUTDIR}
cp ${SCRIPTDIR}/unittest_FU.py ${OUTDIR}
cp ${SCRIPTDIR}/test_dqmstream.py ${OUTDIR}
cd ${OUTDIR}

rm -rf $OUTDIR/{ramdisk,data,dqmdisk,*.log}

runnumber="100101"
echo "Running test with FRD file header (no index JSONs)"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=1"
#CMDLINE_STARTFU="cmsRun startFU.py runNumber=${runnumber} fffBaseDir=${OUTDIR}"
CMDLINE_STARTFU="cmsRun ${FUSCRIPT} runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR

#prepare DQM files
mkdir dqmdisk/run${runnumber} -p
cat data/run${runnumber}/run${runnumber}_ls0000_streamDQM_pid*.ini > dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamDQM_pid*.dat >> dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat
find dqmdisk
echo '{"data": [12950, 1620, 0, "run'${runnumber}'_ls0001_streamDQM_test.dat", 40823782, 1999348078, 135, 13150, 0, "Failsafe"]}' > dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.jsn

CMDLINE_STARTDQM="cmsRun test_dqmstream.py runInputDir=./dqmdisk runNumber=100101"
${CMDLINE_STARTDQM} > out_2_dqm.log 2>&1 || diedqm "${CMDLINE_STARTDQM}" $? $OUTDIR

#no failures, clean up everything including logs if there are no errors
echo "Completed sucessfully"
#rm -rf $OUTDIR/{ramdisk,data,*.py,*.log}
rm -rf $OUTDIR

exit ${RC}
