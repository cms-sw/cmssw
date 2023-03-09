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

TEST_DIR=$CMSSW_BASE/src/RecoMTD/DetLayers/test

F1=${TEST_DIR}/mtd_cfg.py

REF_FILE="Geometry/TestReference/data/mtdDetLayerGeometryRef.log.gz"
REF=""
for d in $(echo $CMSSW_SEARCH_PATH | tr ':' '\n') ; do
  if [ -e "${d}/${REF_FILE}" ] ; then
    REF="${d}/${REF_FILE}"
      break
  fi
done
[ -z $REF ] && exit 1

FILE1=mtdDetLayerGeometry.log
LOG=mtddlglog
DIF=mtddlgdif

echo " testing RecoMTD/DetLayers"

echo "===== Test \"cmsRun mtd_cfg.py\" ===="
rm -f $LOG $DIF $FILE1

cmsRun $F1 >& $LOG || die "Failure using cmsRun $F1" $?
gzip -f $FILE1 || die "$FILE1 compression fail" $?
(zdiff $FILE1.gz $REF >& $DIF || [ -s $DIF ] && checkDiff $DIF || echo "OK") || die "Failure in comparison for $FILE1" $?
