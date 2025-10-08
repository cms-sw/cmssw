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
cp ${SCRIPTDIR}/ufu2.py ${OUTDIR}
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

echo "running DAQSource test with striped event FRD (SFB)"
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2 buBaseDir=ramdisk1 subsystems=TCDS,SiPixel,ECAL,RPC"
#CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2 buBaseDir=ramdisk1 subsystems=TCDS,SiPixel,ECAL,RPC eventDataType=65535"
${CMDLINE_STARTBU}
CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2 buBaseDir=ramdisk2 subsystems=SiStrip,HCAL,DT,CSC eventDataType=65535"
#CMDLINE_STARTBU="cmsRun startBU.py runNumber=${runnumber} fffBaseDir=${OUTDIR} maxLS=2 fedMeanSize=128 eventsPerFile=20 eventsPerLS=35 frdFileVersion=2 buBaseDir=ramdisk2 subsystems=SiStrip,HCAL,DT,CSC"
${CMDLINE_STARTBU}
#run reader
CMDLINE_STARTFU="cmsRun startFU_daqsource.py daqSourceMode=FRDStriped runNumber=${runnumber} fffBaseDir=${OUTDIR} numRamdisks=2"
${CMDLINE_STARTFU}

rm -rf $OUTDIR
exit ${RC}
