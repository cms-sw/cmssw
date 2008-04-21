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
cd $HOME/scratch0/CMSSW_1_7_5
echo Setting up $(pwd) as CMSSW environment. 
eval `scramv1 runtime -sh`
rehash

cd $BATCH_DIR
echo The running directory is $(pwd).
# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

gzip -f *.log
echo "\nDirectory content after running cmsRun and zipping log file:"
ls -lh 
# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space:
cp -p *.log.gz *.root $RUNDIR

# Copy MillePede binary file to Castor,
# so first set castor pool for binary files in $MSSDIR area:
export STAGE_SVCCLASS=$MSSDIRPOOL
nsrm -f $MSSDIR/milleBinaryISN.dat
echo 'rfcp milleBinaryISN.dat $MSSDIR/'
rfcp milleBinaryISN.dat $MSSDIR/
