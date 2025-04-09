#!/bin/bash
SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function diebu {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_bu.log;  rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }
function diefu {  echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_fu.log;  rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }
function diedqm { echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_dqm.log; rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }
function dieecal { echo Failure $1: status $2 ; echo "" ; echo "----- Error -----"; echo ""; cat out_2_ecal.log; rm -rf $3/{ramdisk,data,dqmdisk,ecalInDir,*.py}; exit $2 ; }

copy_index_files() {
  directory=$1
  sourceid=$2
  del_orig=$3
  shopt -s nullglob
  for file in "$directory"/*_index*.raw; do
    filename=$(basename "$file")
    if [[ "$filename" =~ ^(.*)_index([0-9]+)\.raw$ ]]; then
        base="${BASH_REMATCH[1]}"
        x="${BASH_REMATCH[2]}"
        new_name="${base}_index${x}_source${sourceid}.raw"
        cp -- "$file" "$directory/$new_name"
        #echo "Copied: $filename -> $new_name"
        if [[ $del_orig -eq 1 ]]; then
          rm -rf $file
        fi
    fi
  done
  shopt -u nullglob
}

copy_json_files() {
  directory=$1
  sourceid=$2
  shopt -s nullglob
  for file in "$directory"/*.jsn; do
    filename=$(basename "$file")
    if [[ "$filename" =~ ^(.*)_EoR.jsn$ ]]; then
        base="${BASH_REMATCH[1]}"
        x="${BASH_REMATCH[2]}"
        new_name="${base}_EoR_source${sourceid}.jsn"
        mv "$file" "$directory/$new_name"
    fi
    if [[ "$filename" =~ ^(.*)_EoLS.jsn$ ]]; then
        base="${BASH_REMATCH[1]}"
        x="${BASH_REMATCH[2]}"
        new_name="${base}_EoLS_source${sourceid}.jsn"
        mv "$file" "$directory/$new_name"
    fi
  done
  shopt -u nullglob
}

FUSCRIPT="startFU.py"

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
cp ${SCRIPTDIR}/startFU_daqsource.py ${OUTDIR}
cp ${SCRIPTDIR}/unittest_FU_daqsource.py ${OUTDIR}
cp ${SCRIPTDIR}/startFU_ds_multi.py ${OUTDIR}
cp ${SCRIPTDIR}/test_dqmstream.py ${OUTDIR}
cp ${SCRIPTDIR}/testECALCalib_cfg.py ${OUTDIR}
cd ${OUTDIR}

rm -rf $OUTDIR/{ramdisk,data,dqmdisk,ecalInDir,*.log}

runnumber="100101"

################
echo "Running fileListMode test"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2"
CMDLINE_STARTFU="cmsRun unittest_FU.py runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR

rm -rf $OUTDIR/{ramdisk,data,*.log}

echo "Running test with FRD file header v1 (no index JSONs)"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=40 eventsPerLS=55 frdFileVersion=1"
#CMDLINE_STARTFU="cmsRun startFU.py runNumber=${runnumber} fffBaseDir=${OUTDIR}"
CMDLINE_STARTFU="cmsRun ${FUSCRIPT} runNumber=${runnumber} fffBaseDir=${OUTDIR}"
mkdir dqmdisk/run${runnumber} -p
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR

#prepare DQM and ECAL Calibration files
cat data/run${runnumber}/run${runnumber}_ls0000_streamDQM_pid*.ini > dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamDQM_pid*.dat >> dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat

rm -rf $OUTDIR/{data}
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR

#prepare DQM and ECAL Calibration files, merged from two processes, containing two metadata events
cat data/run${runnumber}/run${runnumber}_ls0001_streamDQM_pid*.dat >> dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat

find dqmdisk
echo '{"data": [12950, 1620, 0, "run'${runnumber}'_ls0001_streamDQM_test.dat", 40823782, 1999348078, 135, 13150, 0, "Failsafe"]}' > dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.jsn

mkdir ecalInDir
cp dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat ecalInDir/

echo "Running DQM source"
CMDLINE_STARTDQM="cmsRun test_dqmstream.py runInputDir=./dqmdisk runNumber=100101 maxLS=1 eventsPerLS=35"
${CMDLINE_STARTDQM} > out_2_dqm.log 2>&1 || diedqm "${CMDLINE_STARTDQM}" $? $OUTDIR

echo "Running ECAL Calibration source"
CMDLINE_STARTECAL="cmsRun testECALCalib_cfg.py"
${CMDLINE_STARTECAL} > out_2_ecal.log 2>&1 || dieecal "${CMDLINE_STARTECAL}" $? $OUTDIR


rm -rf $OUTDIR/{ramdisk,data,dqmdisk,ecalInDir,*.log}

###################
echo "Running test with FRD file header v1 (no index JSONs) and empty files"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=1"
#CMDLINE_STARTFU="cmsRun startFU.py runNumber=${runnumber} fffBaseDir=${OUTDIR}"
CMDLINE_STARTFU="cmsRun ${FUSCRIPT} runNumber=${runnumber} fffBaseDir=${OUTDIR} numEventsToWrite=0"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR

#prepare DQM files
mkdir dqmdisk/run${runnumber} -p
cat data/run${runnumber}/run${runnumber}_ls0000_streamDQM_pid*.ini > dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat
cat data/run${runnumber}/run${runnumber}_ls0001_streamDQM_pid*.dat >> dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.dat
find dqmdisk
echo '{"data": [12950, 1620, 0, "run'${runnumber}'_ls0001_streamDQM_test.dat", 40823782, 1999348078, 135, 13150, 0, "Failsafe"]}' > dqmdisk/run${runnumber}/run${runnumber}_ls0001_streamDQM_test.jsn

echo "Running DQM source"
CMDLINE_STARTDQM="cmsRun test_dqmstream.py runInputDir=./dqmdisk runNumber=100101 maxLS=1 eventsPerLS=0"
${CMDLINE_STARTDQM} > out_2_dqm.log 2>&1 || diedqm "${CMDLINE_STARTDQM}" $? $OUTDIR

rm -rf $OUTDIR/{ramdisk,data,dqmdisk,*.log}

################
echo "Running test with FRD file header v2"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2"
CMDLINE_STARTFU="cmsRun ${FUSCRIPT} runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR

rm -rf $OUTDIR/{ramdisk,data,*.log}

echo "running DAQSource fileListMode test with full event FRD"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=1"
CMDLINE_STARTFU="cmsRun unittest_FU_daqsource.py daqSourceMode=FRD runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR out_2_fu.log

#no failures, clean up everything including logs if there are no errors
rm -rf $OUTDIR/{ramdisk,data,*.log}

echo "running DAQSource test with striped event FRD"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2 buBaseDir=ramdisk1 subsystems=TCDS,SiPixel,ECAL,RPC"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2 buBaseDir=ramdisk2 subsystems=SiStrip,HCAL,DT,CSC"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
#run reader
CMDLINE_STARTFU="cmsRun startFU_daqsource.py daqSourceMode=FRDStriped runNumber=${runnumber} fffBaseDir=${OUTDIR} numRamdisks=2"
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR out_2_fu.log
rm -rf $OUTDIR/{ramdisk,data,*.log}

echo "running DAQSource test with FRDPreUnpack"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=1"
CMDLINE_STARTFU="cmsRun startFU_daqsource.py daqSourceMode=FRDPreUnpack runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR out_2_fu.log

#no failures, clean up everything including logs if there are no errors
rm -rf $OUTDIR/{ramdisk,data,*.log}

echo "running DAQSource test with raw DTH orbit payload"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=2 eventsPerLS=3 frdFileVersion=0 dataType=DTH"
CMDLINE_STARTFU="cmsRun startFU_daqsource.py daqSourceMode=DTH runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR out_2_fu.log

#no failures, clean up everything including logs if there are no errors
rm -rf $OUTDIR/{ramdisk,data,*.log}

echo "running DAQSource test with striped DTH"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=2 eventsPerLS=3 frdFileVersion=0 dataType=DTH"
CMDLINE_STARTFU="cmsRun startFU_ds_multi.py daqSourceMode=DTH runNumber=${runnumber} fffBaseDir=${OUTDIR}"
${CMDLINE_STARTBU}  > out_2_bu.log 2>&1 || diebu "${CMDLINE_STARTBU}" $? $OUTDIR
#duplicate files
copy_index_files ramdisk/run${runnumber} 0111
copy_index_files ramdisk/run${runnumber} 0222 1
copy_json_files ramdisk/run${runnumber} 0111
#find ramdisk/run${runnumber}
${CMDLINE_STARTFU}  > out_2_fu.log 2>&1 || diefu "${CMDLINE_STARTFU}" $? $OUTDIR out_2_fu.log

rm -rf $OUTDIR/{ramdisk,data,*.log}

#no failures, clean up everything including logs if there are no errors
echo "Completed sucessfully"
#rm -rf $OUTDIR/{ramdisk,data,*.py,*.log}
rm -rf $OUTDIR

exit ${RC}
