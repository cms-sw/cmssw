source @root/scripts/setup.sh -j RunInfoStopTest
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_test_command RunInfoStopTest "cmsRun $SRCDIR/RunInfoPopConAnalyzer_cfg.py runNumber=$1 destinationConnection={db} tag={tag}"
