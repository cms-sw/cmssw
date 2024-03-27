source /data/O2O/scripts/setupO2O.sh -s RunInfo -j StartTest
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_test_command RunInfoStartTest "cmsRun $SRCDIR/RunInfoPopConAnalyzer_cfg.py runNumber=$1 destinationConnection={db} tag={tag}"
