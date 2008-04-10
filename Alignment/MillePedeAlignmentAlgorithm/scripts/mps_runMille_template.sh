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

# set up the CMS environment
cd $HOME/scratch0/CMSSW_1_7_5
eval `scramv1 runtime -sh`
rehash

cd $RUNDIR
echo Running directory changed to $(pwd).
rm -f alignment.log milleBinary.dat millePedeMonitor.root treeFile.root histograms.root LSFJOB STDOUT

# create and symlink temporary binary file into run directory
TMP_BINARY=$BATCH_DIR/milleBinaryISN.dat
touch $TMP_BINARY
ln -s $TMP_BINARY milleBinary.dat

echo The running directory is $(pwd)
# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

# set castor pool for binary files in $MSSDIR area
export STAGE_SVCCLASS=$MSSDIRPOOL

# copy MillePede binary file to Castor
nsrm -f $MSSDIR/milleBinaryISN.dat
echo 'rfcp $TMP_BINARY $MSSDIR/'
rfcp $TMP_BINARY $MSSDIR/

# clean up disc (=> Not really necessary, job directory will anyway vanish with job!...)
rm $TMP_BINARY
# clean up dangling link
rm milleBinary.dat
