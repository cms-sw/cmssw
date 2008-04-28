#!/bin/zsh 
#
# Run script template for Pede job, copying (rfcp) binary files from mass storage to local disk.
#
# Adjustments might be needed for CMSSW environment.

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
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

cd $BATCH_DIR
echo Running directory changed to $(pwd).

# create link for treeFile(s) in mille job $RUNDIR's
# (comment in case you a cfg not creating treeFiles...)
ln -s $RUNDIR/../jobISN/treeFile.root treeFileISN.root

# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

# clean the link created above to avoid copying later (maybe uncomment, see above)
rm  treeFileISN.root

gzip -f *.log *.txt

# Merge possible alignment monitor and millepede monitor hists...
# ...and remove individual histogram files after merging to save space (if
success):
#NOTE: the names "histograms.root" and "millePedeMonitor.root" must match
what is in
#      the mps_template.cfg!
hadd histograms_merge.root $RUNDIR/../job???/histograms.root
if [ $? -eq 0 ]; then
    rm $RUNDIR/../job???/histograms.root
fi
hadd millePedeMonitor_merge.root $RUNDIR/../job???/millePedeMonitor.root
if [ $? -eq 0 ]; then
    rm $RUNDIR/../job???/millePedeMonitor.root
fi

echo "\nDirectory content after running cmsRun, zipping log file and merging
histogram files:"
ls -lh 
# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space:
cp -p *.log.gz *.txt.gz *.root millepede.*s $RUNDIR
