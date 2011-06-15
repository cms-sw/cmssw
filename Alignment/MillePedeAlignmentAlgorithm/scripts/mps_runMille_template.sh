#!/bin/zsh 
#
# Run script template for Mille jobs
#
# Adjustments might be needed for CMSSW environment.
#
# In the very beginning of this script, stager requests for the files will be added.

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
MSSDIRPOOL=

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

# set up the CMS environment (choose your release and working area):
cd CMSSW_RELEASE_AREA
echo Setting up $(pwd) as CMSSW environment. 
eval `scram runtime -sh`
rehash

cd $BATCH_DIR
echo The running directory is $(pwd).
# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

gzip -f *.log
gzip milleBinaryISN.dat
echo "\nDirectory content after running cmsRun and zipping log+dat files:"
ls -lh 
# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.log.gz $RUNDIR
# store  millePedeMonitor also in $RUNDIR, below is backup in $MSSDIR
cp -p millePedeMonitor*root $RUNDIR

# Copy MillePede binary file to Castor
# Must use different command for the cmscafuser pool
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
# Not using cmscafuser pool => rfcp command must be used
  export STAGE_SVCCLASS=$MSSDIRPOOL
  export STAGER_TRACE=
  nsrm -f $MSSDIR/milleBinaryISN.dat.gz
  echo "rfcp milleBinaryISN.dat.gz $MSSDIR/"
  rfcp milleBinaryISN.dat.gz    $MSSDIR/
  rfcp treeFile*root         $MSSDIR/treeFileISN.root
  rfcp millePedeMonitor*root $MSSDIR/millePedeMonitorISN.root
else
# Using cmscafuser pool => cmsStageOut command must be used
  . /afs/cern.ch/cms/caf/setup.sh
  MSSCAFDIR=`echo $MSSDIR | awk 'sub("/castor/cern.ch/cms","")'`
  echo "cmsStageOut milleBinaryISN.dat.gz $MSSCAFDIR/milleBinaryISN.dat.gz > /dev/null"
  cmsStageOut milleBinaryISN.dat.gz    $MSSCAFDIR/milleBinaryISN.dat.gz  > /dev/null
  cmsStageOut treeFile*root         $MSSCAFDIR/treeFileISN.root > /dev/null
  cmsStageOut millePedeMonitor*root $MSSCAFDIR/millePedeMonitorISN.root > /dev/null
fi
