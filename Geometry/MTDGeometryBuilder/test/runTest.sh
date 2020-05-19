#!/bin/sh

function die { echo $1: status $2 ; exit $2; }
function checkDiff {
    FSIZE=$(stat -c%s "$1")
    echo "The output diff is $FSIZE:"
    cat $1;
    if [ $FSIZE -gt 4331 ]
    then
      exit -1;
    fi
}

F1=${LOCAL_TEST_DIR}/mtd_cfg.py
F2=${LOCAL_TEST_DIR}/dd4hep_mtd_cfg.py
REF=${LOCAL_TOP_DIR}/src/Geometry/TestReference/data/mtdGeometryRef.log.gz
FILE1=mtdGeometryDDD.log
FILE2=mtdGeometryDD4hep.log
LOG=mtdgblog
DIF=mtdgbdif

echo " testing Geometry/MTDGeometryBuilder"

export tmpdir=${LOCAL_TMP_DIR:-/tmp}
echo "===== Test \"cmsRun mtd_cfg.py\" ===="
rm -f $LOG $DIF $FILE1
(cmsRun $F1 >& $LOG;
    gzip -f $FILE1; zdiff $FILE1.gz $REF >& $DIF;
    [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure using cmsRun $F1" $?
rm -f $LOG $DIF $FILE2
echo "===== Test \"cmsRun dd4hep_mtd_cfg.py\" ===="
(cmsRun $F2 >& $LOG;
    gzip -f $FILE2; zdiff $FILE2.gz $REF >& $DIF;
    [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure using cmsRun $F2" $?
