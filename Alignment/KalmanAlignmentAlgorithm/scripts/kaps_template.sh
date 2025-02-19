#!/bin/zsh 
#
# This script is part of the Kalman Alignment Production System (KAPS).
# It is a script template for an alignment job.
#
# Adjustments might be needed for CMSSW environment.
#

# these defaults will be overwritten by MPS
RUNDIR=rundir
MSSDIR=mssdir

MSSDIRPOOL=cmscaf
export STAGE_SVCCLASS=$MSSDIRPOOL

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

# set up the CMS environment (choose your release and working area):
cd /afs/hephy.at/scratch/e/ewidl/CMSSW_2_1_10
echo Setting up $(pwd) as CMSSW environment. 
eval `scramv1 runtime -sh`
rehash

cd $BATCH_DIR
rm -f alignmentISN.log kaaOutputISN.root debugISN.root

echo The running directory is $(pwd).
# Execute. The cfg file name will be overwritten by MPS
time cmsRun the_cfg.py

echo "\nDirectory content after running cmsRun:"
ls -lh 

# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space:
gzip alignmentISN.log
cp -p alignmentISN.log.gz kaaDebugISN.root $RUNDIR

# Copy output file to the dedicated AFS directory
rm -f $MSSDIR/kaaOutputISN.root
echo 'cp kaaOutputISN.root $MSSDIR/'
cp kaaOutputISN.root $MSSDIR/
