#!/bin/zsh 
#
# This script is part of the Kalman Alignment Production System (KAPS).
# It is a script template for merge jobs.
#
# Adjustments might be needed for CMSSW environment.
#

# these defaults will be overwritten by MPS
RUNDIR=rundir
MSSDIR=mssdir
MSSDIRPOOL=

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

cp $MSSDIR/kaaOutputISN.root $BATCH_DIR

# set up the CMS environment
cd /afs/hephy.at/scratch/e/ewidl/CMSSW_2_1_10
echo Setting up $(pwd) as CMSSW environment. 
eval `scramv1 runtime -sh`
rehash

cd $BATCH_DIR
echo Running directory changed to $(pwd).

# Execute. The cfg file name will be overwritten by MPS
time cmsRun merge_cfg.py

echo "\nDirectory content after running cmsRun"
ls -lh
# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space:

cp -p alignment.log $RUNDIR
cp -p kaaMerged.root $MSSDIR
