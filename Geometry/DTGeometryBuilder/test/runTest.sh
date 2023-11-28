#!/bin/sh

function die { echo $1: status $2 ; exit $2; }
function checkDiff {
    FSIZE=$(stat -c%s "$1")
    echo "The output diff is $FSIZE:"
    cat $1;
    if [ $FSIZE -gt 2000 ]
    then
	exit -1;
    fi
}

F1=${SCRAM_TEST_PATH}/python/validateDTGeometry_cfg.py
F2=${SCRAM_TEST_PATH}/python/testDTGeometry.py
FILE1=${SCRAM_TEST_PATH}/dtGeometry.log.org
FILE2=dtGeometry.log
FILE3=diff.log
FILE4=dtGeometryFiltered.log

echo " testing Geometry/DTGeometryBuilder"

export tmpdir=${PWD}
# The following test does not work with DD4hep with Geant4 units
# echo "===== Test \"cmsRun validateDTGeometry_cfg.py\" ===="
# (cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testDTGeometry.py\" ===="
(cmsRun $F2;
    grep -v 'Benchmark ' $FILE2 | grep -v '^ *[1-9]' | grep -v '%MSG-i' | grep -v '^Info '>& $FILE4;
    diff -B -w $FILE1 $FILE4 >& $FILE3;
    [ -s $FILE3 ] && checkDiff $FILE3 || echo "OK") || die "Failure using cmsRun $F2" $?
