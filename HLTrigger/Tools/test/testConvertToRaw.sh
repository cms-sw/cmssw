#!/bin/bash

# Pass in name and status
function die {
  printf "\n%s: status %s\n" "$1" "$2"
  exit $2
}

check_for_success() {
    "${@}" && echo -e "\n ---> Passed test of '${@}'\n\n" || exit 1
}

check_for_failure() {
    "${@}" && exit 1 || echo -e "\n ---> Passed test of '${@}'\n\n"
}

inputfile="/store/data/Run2024C/EphemeralHLTPhysics0/RAW/v1/000/379/416/00000/e8dd5e3c-216f-4545-acb6-ab86c9161085.root"

echo "============================================================"
echo "Testing convertToRaw in ${SCRAM_TEST_PATH}."
echo "------------------------------------------------------------"
echo

echo "============================================================"
echo "testing help function "
echo "------------------------------------------------------------"
echo

convertToRaw --help  || die "Failure running convertToRaw --help" $?

echo "============================================================"
echo "testing successful conversion of edm files"
echo "------------------------------------------------------------"
echo

check_for_success convertToRaw -f 1 -l=1 -v $inputfile

echo "============================================================"
echo "testing failing conversion of edm files"
echo "------------------------------------------------------------"
echo

check_for_failure convertToRaw -f 1 -l=-1 -s rawDataRepacker $inputfile

echo "============================================================"
echo "generating a streamer file"
echo "------------------------------------------------------------"
echo
cat > stream.py << @EOF
import FWCore.ParameterSet.Config as cms

process = cms.Process( "STREAMER" )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('$inputfile')
)

process.maxEvents.input = 100

from EventFilter.Utilities.EvFDaqDirector_cfi import EvFDaqDirector as _EvFDaqDirector
process.EvFDaqDirector = _EvFDaqDirector.clone(
    baseDir = "$PWD",
    buBaseDir = "$PWD",
    runNumber = 379416
)

process.FastMonitoringService = cms.Service("FastMonitoringService")

process.hltOutputTest = cms.OutputModule( "GlobalEvFOutputModule",
    use_compression = cms.untracked.bool( True ),
    compression_algorithm = cms.untracked.string( "ZSTD" ),
    compression_level = cms.untracked.int32( 3 ),
    outputCommands = cms.untracked.vstring( 'keep *')
)

process.out = cms.EndPath(process.hltOutputTest)
@EOF

mkdir run379416
cmsRun stream.py
cat run379416/run379416_ls0000_streamTest_pid*.ini run379416/run379416_ls0097_streamTest_pid*.dat > run379416_ls0097_streamTest.dat
rm -rf run379416

echo "============================================================"
echo "testing successful conversion of streamer files"
echo "------------------------------------------------------------"
echo

check_for_success convertToRaw -f 1 -l=1 -v run379416_ls0097_streamTest.dat

echo "============================================================"
echo "testing failing conversion of streamer files"
echo "------------------------------------------------------------"
echo

check_for_failure convertToRaw -f 1 -l=-1 -s rawDataRepacker run379416_ls0097_streamTest.dat
