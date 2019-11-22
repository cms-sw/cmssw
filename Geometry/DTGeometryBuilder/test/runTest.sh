#!/bin/sh

function die { echo $1: status $2 ; exit $2; }
function checkDiff {
    FSIZE=$(stat -c%s "$1")
    echo "The output diff is $FSIZE:"
    cat $1;
    if [ $FSIZE -gt 0 ]
    then
	exit -1;
    fi
}

F1=${LOCAL_TEST_DIR}/python/validateDTGeometry_cfg.py
F2=${LOCAL_TEST_DIR}/python/testDTGeometry.py
FILE1=${LOCAL_TEST_DIR}/dtGeometry.log.org
FILE2=dtGeometry.log
FILE3=diff.log

echo " testing Geometry/DTGeometryBuilder"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun validateDTGeometry_cfg.py\" ===="
(cmsRun $F1) || die "Failure using cmsRun $F1" $?
echo "===== Test \"cmsRun testDTGeometry.py\" ===="
(cmsRun $F2;
    diff $FILE1 $FILE2 >& $FILE3;
    [ -s $FILE3 ] && checkDiff $FILE3 || echo "OK") || die "Failure using cmsRun $F2" $?
