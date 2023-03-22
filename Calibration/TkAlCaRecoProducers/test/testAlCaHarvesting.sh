#! /bin/bash

function die { echo $1: status $2 ; exit $2; }
function cleanTheHouse {
    rm -fr millepede.*
    rm -fr pede*
    rm -fr treeFile.root
    rm -fr testPCLAlCaHarvesting.db
    rm -fr HGalignment
}

echo "TESTING Calibration/TkAlCaRecoProducers ..."
cmsRun -e -j testPCLAlCaHarvesting.xml ${SCRAM_TEST_PATH}/testPCLAlCaHarvesting.py || die "Failure running testPCLAlCaHarvesting.py" $?
cleanTheHouse
echo "PARSING Framework Job Report ..."
python3 ${SCRAM_TEST_PATH}/parseFwkJobReport.py
