source /data/O2O/scripts/setupO2O.sh -s RunInfo -j Start
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_command RunInfoStart "cmsRun $SRCDIR/RunInfoPopConAnalyzer.py runNumber=$1 destinationConnection={db} tag={tag}"
