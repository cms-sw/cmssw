#!/bin/bash
function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

if test -f "milleBinary00*"; then
    echo "cleaning the local test area"
    rm -fr milleBinary00* 
    rm -fr pedeSteer* 
fi

pwd
echo " testing Aligment/MillePedeAlignmentAlgorithm"

REMOTE="/store/group/alca_global/tkal_millepede_tests/"
TESTPACKAGE="test_pede_package.tar"
COMMMAND=`xrdfs cms-xrd-global.cern.ch locate ${REMOTE}${TESTPACKAGE}`
STATUS=$?
echo "xrdfs command status = "$STATUS
if [ $STATUS -eq 0 ]; then
    echo "Using file ${TESTPACKAGE}. Running in ${LOCAL_TEST_DIR}."
    xrdcp root://cms-xrd-global.cern.ch/${REMOTE}${TESTPACKAGE} ${LOCAL_TEST_DIR}
    tar -xvf ${LOCAL_TEST_DIR}/${TESTPACKAGE} 
    gunzip milleBinary00*    
    (cmsRun ${LOCAL_TEST_DIR}/test_pede.py) || die 'failed running test_pede.py' $?
    echo -e "\n MillePede Exit Status: "`cat millepede.end`
else 
  die "SKIPPING test, file ${TESTPACKAGE} not found" 0
fi
