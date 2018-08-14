source @root/scripts/setup.sh -j RunInfoStart
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_command RunInfoStart "cmsRun $SRCDIR/RunInfoPopConAnalyzer.py runNumber=$1 destinationConnection={db} tag={tag}"
