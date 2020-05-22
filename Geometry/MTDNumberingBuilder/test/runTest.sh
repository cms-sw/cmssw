#!/bin/sh

function die { echo $1: status $2 ; exit $2; }
function checkDiff {
    FSIZE=$(stat -c%s "$1")
    echo "The output diff is $FSIZE:"
    cat $1;
    if [ $FSIZE -gt 5216 ]
    then
      exit -1;
    fi
}

F1=${CMSSW_BASE}/src/Geometry/MTDNumberingBuilder/test/mtd_cfg.py
F2=${CMSSW_BASE}/src/Geometry/MTDNumberingBuilder/test/dd4hep_mtd_cfg.py
REF=${CMSSW_BASE}/src/Geometry/TestReference/data/mtdNumberingRef.log.gz
FILE1=mtdNumberingDDD.log
FILE2=mtdNumberingDD4hep.log
LOG=mtdnblog
DIF=mtdnbdif

echo " testing Geometry/MTDNumberingBuilder"

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
