#!/bin/bash
function die { echo $1: status $2; exit $2; }

LOCAL_TEST_DIR=${SCRAM_TEST_PATH}

clean_up(){
    echo "cleaning the local test area"
    rm -fr milleBinary00* 
    rm -fr pedeSteer* 
    rm -fr millepede.*
    rm -fr *.root
    rm -fr *.log
    rm -fr *.dat
    rm -fr *.tar
    rm -fr *.gz
    rm -fr *.db
}

if test -f "milleBinary*"; then
    clean_up
fi

pwd
echo " testing Aligment/MillePedeAlignmentAlgorithm"

REMOTE="/store/group/alca_global/tkal_millepede_tests/"
TESTPACKAGE="test_pede_package_v1"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate ${REMOTE}${TESTPACKAGE}.tar`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${TESTPACKAGE}. Running in ${LOCAL_TEST_DIR}."
    xrdcp root://cms-xrd-global.cern.ch/${REMOTE}${TESTPACKAGE}.tar .
    tar -xvf ${TESTPACKAGE}.tar
    mv ${TESTPACKAGE}/milleBinary* .
    mv ${TESTPACKAGE}/alignment_input.db .
    gunzip milleBinary*
    (cmsRun ${LOCAL_TEST_DIR}/test_pede.py) || die 'failed running test_pede.py' $?
    echo -e "\n @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo -e " @ MillePede Exit Status: "`cat millepede.end`
    echo -e " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    ## mv the output file to the local test directory for the subsequent payload sanity check
    mv alignments_MP.db ${LOCAL_TEST_DIR}
    ## clean the house now...
    clean_up
else 
  die "SKIPPING test, file ${TESTPACKAGE}.tar not found" 0
fi
