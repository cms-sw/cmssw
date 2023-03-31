#!/bin/sh

function die { echo $1: status $2 ; exit $2; }
function checkDiff {
    FSIZE=$(stat -c%s "$1")
    echo "The output diff is $FSIZE:"
    cat $1;
    if [ $FSIZE -gt 500 ]
    then
	exit -1;
    fi
}

echo " testing Geometry/TrackerGeomtryBuilder"

for entry in "${SCRAM_TEST_PATH}/python"/test*
do
  echo "===== Test \"cmsRun $entry \" ===="
  (cmsRun $entry) || die "Failure using cmsRun $entry" $?
done

cmsRun ${SCRAM_TEST_PATH}/python/testPixelTopologyMapTest_cfg.py runNumber=300000  || die "Failure using cmsRun testPixelTopologyMapTest_cfg.py runNumber=300000" $?
cmsRun ${SCRAM_TEST_PATH}/python/testPixelTopologyMapTest_cfg.py globalTag=auto:phase2_realistic_T21  || die "Failure using cmsRun testPixelTopologyMapTest_cfg.py globalTag=auto:phase2_realistic_T21" $?

FILE1=trackerParametersDD4hep.log
FILE2=trackerParametersDDD.log
FILE3=diff.log

echo "===== Compare Tracker Parameters for DD and DD4hep ===="
(diff -B -w $FILE1 $FILE2 >& $FILE3;
    [ -s $FILE3 ] && checkDiff $FILE3 || echo "OK") || die "Failure comparing Tracker Parameters for DD and DD4hep" $?
    
