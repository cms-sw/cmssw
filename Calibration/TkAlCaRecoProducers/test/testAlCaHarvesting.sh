#! /bin/bash

function die { echo $1: status $2 ; exit $2; }
function cleanTheHouse {
    rm -fr millepede.*
    rm -fr pede*
    rm -fr treeFile.root
}

echo "TESTING Calibration/TkAlCaRecoProducers ..."
cmsRun -e ${LOCAL_TEST_DIR}/testPCLAlCaHarvesting.py || die "Failure running testPCLAlCaHarvesting.py" $? 
cleanTheHouse
echo "PARSING Framework Job Report ..."
python ${LOCAL_TEST_DIR}/parseFwkJobReport.py
