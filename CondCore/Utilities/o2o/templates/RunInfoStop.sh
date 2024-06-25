source @root/scripts/setup.sh -j RunInfoStop
SRCDIR=$RELEASEDIR/src/CondTools/RunInfo/python
submit_command RunInfoStop "cmsRun $SRCDIR/RunInfoPopConAnalyzer_cfg.py runNumber=$1 destinationConnection={db} tag={tag}"
