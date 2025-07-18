#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function diebu {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_bu.log;  rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }
function diefu {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_fu.log;  rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }
function diedqm { echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_dqm.log; rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }
function dieecal { echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_ecal.log; rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }

FUSCRIPT="unittest_FU.py"
if [ ! -z $1 ]; then
  if [ "$1" == "local" ]; then
    FUSCRIPT="startFU.py"
    echo "local run: using ${FUSCRIPT}"
  fi
fi

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
cp ${SCRIPTDIR}/startBU.py ${OUTDIR}
cp ${SCRIPTDIR}/startFU.py ${OUTDIR}
cp ${SCRIPTDIR}/unittest_FU.py ${OUTDIR}
cp ${SCRIPTDIR}/unittest_FU_daqsource.py ${OUTDIR}
cp ${SCRIPTDIR}/test_dqmstream.py ${OUTDIR}
cp ${SCRIPTDIR}/testECALCalib_cfg.py ${OUTDIR}
cd ${OUTDIR}

rm -rf $OUTDIR/{ramdisk,data,dqmdisk,ecalInDir,*.log}

runnumber="100101"

echo "running DAQSource test with raw DTH orbits"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=2 eventsPerLS=3 frdFileVersion=0 dataType=DTH"
CMDLINE_STARTFU="cmsRun unittest_FU_daqsource.py daqSourceMode=DTH runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR out_2_fu.log

#no failures, clean up everything including logs if there are no errors
rm -rf $OUTDIR
exit 0

#######################################################################
