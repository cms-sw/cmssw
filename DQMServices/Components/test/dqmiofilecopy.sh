#!/bin/bash
# like '/SingleElectron/Run2017D-09Aug2019_UL2017-v1/DQMIO'
DATASET=$1
# like 302663
RUN=$2

SOURCE='root://cms-xrd-global.cern.ch/'

DIR=$(echo $DATASET | tr / _)

FILES=$(dasgoclient -query="file dataset=$DATASET run=$RUN" -limit=-1)
mkdir $DIR

echo 'process.source.fileNames = cms.untracked.vstring('
for f in $FILES; do
  edmCopyUtil "$SOURCE$f" $DIR &
  echo "  'file:$DIR/$(basename $f)',"
done
echo ')'

# wait for parallel transfers to complete
wait





