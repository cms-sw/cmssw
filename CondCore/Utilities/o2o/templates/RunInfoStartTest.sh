source @root/scripts/setup.sh -j RunInfoStartTest
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_test_command RunInfoStartTest "cmsRun $SRCDIR/RunInfoPopConAnalyzer.py runNumber=$1 destinationConnection={db} tag={tag}"
