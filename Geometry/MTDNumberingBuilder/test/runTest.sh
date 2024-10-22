#!/bin/sh -e

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

TEST_DIR=$CMSSW_BASE/src/Geometry/MTDNumberingBuilder/test

F1=${TEST_DIR}/mtd_cfg.py
F2=${TEST_DIR}/dd4hep_mtd_cfg.py

REF_FILE="Geometry/TestReference/data/mtdNumberingRef.log.gz"
REF=""
for d in $(echo $CMSSW_SEARCH_PATH | tr ':' '\n') ; do
  if [ -e "${d}/${REF_FILE}" ] ; then
    REF="${d}/${REF_FILE}"
      break
  fi
done
[ -z $REF ] && exit 1

FILE1=mtdNumberingDDD.log
FILE2=mtdNumberingDD4hep.log
LOG=mtdnblog
DIF=mtdnbdif

echo " testing Geometry/MTDNumberingBuilder"

echo "===== Test \"cmsRun mtd_cfg.py\" ===="
rm -f $LOG $DIF $FILE1

cmsRun $F1 >& $LOG || die "Failure using cmsRun $F1" $?
gzip -f $FILE1 || die "$FILE1 compression fail" $?
(zdiff $FILE1.gz $REF >& $DIF || [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure in comparison for $FILE1" $?

rm -f $LOG $DIF $FILE2
echo "===== Test \"cmsRun dd4hep_mtd_cfg.py\" ===="

cmsRun $F2 >& $LOG || die "Failure using cmsRun $F2" $?
gzip -f $FILE2 || die "$FILE2 compression fail" $?
(zdiff $FILE2.gz $REF >& $DIF || [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure in comparison for $FILE2" $?

