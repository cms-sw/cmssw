#!/bin/zsh 
#
# Run script template for Pede job, copying (rfcp) binary files from mass storage to local disk.
#
# Adjustments might be needed for CMSSW environment.

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/mpede-test/rundir
MSSDIR=/castor/cern.ch/user/r/rmankel/ZMuMu
MSSDIRPOOL=

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

# stage and copy the binary file(s), first set castor pool for binary files in $MSSDIR area
export STAGE_SVCCLASS=$MSSDIRPOOL
stager_get -M $MSSDIR/milleBinaryISN.dat
rfcp $MSSDIR/milleBinaryISN.dat $BATCH_DIR

# set up the CMS environment
cd $HOME/scratch0/CMSSW_1_7_5
eval `scramv1 runtime -sh`
rehash

cd $RUNDIR
echo Running directory changed to $(pwd).
# symlink temporary binary file(s) into run directory
ln -s $BATCH_DIR/milleBinaryISN.dat

echo "rm -f" the following files
#ls mille* pede* treeFile* histograms*
#rm -f mille* pede* treeFile* histograms*
ls treeFile_merge.root histograms_merge.root
rm -f treeFile_merge.root histograms_merge.root

# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

# Merge possible alignment monitor and millepede monitor hists...
# ...and remove individual histogram files after merging to save space (if success):
hadd histograms_merge.root $RUNDIR/../job???/histograms.root
if [ $? -eq 0 ]; then
    rm $RUNDIR/../job???/histograms.root
fi 
hadd millePedeMonitor_merge.root $RUNDIR/../job???/millePedeMonitor.root
if [ $? -eq 0 ]; then
    rm $RUNDIR/../job???/millePedeMonitor.root
fi

# clean up disc and dangling link(s)
rm $BATCH_DIR/milleBinaryISN.dat 
rm milleBinaryISN.dat


