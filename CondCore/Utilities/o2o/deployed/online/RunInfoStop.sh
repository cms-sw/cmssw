source /data/O2O/scripts/setupO2O.sh -s RunInfo -j Stop
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_command RunInfoStop "cmsRun $SRCDIR/RunInfoPopConAnalyzer.py runNumber=$1 destinationConnection={db} tag={tag}"
