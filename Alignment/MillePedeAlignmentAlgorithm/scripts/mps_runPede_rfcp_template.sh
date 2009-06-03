#!/bin/zsh 
#
# Run script template for Pede job, copying binary files from mass storage to local disk.
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
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
# Not using cmscafuser pool => rfcp command must be used
  stager_get -M $MSSDIR/milleBinaryISN.dat
  rfcp $MSSDIR/milleBinaryISN.dat $BATCH_DIR
else
# Using cmscafuser pool => cmsStageIn command must be used
  . /afs/cern.ch/cms/caf/setup.sh
  MSSCAFDIR=`echo $MSSDIR | awk 'sub("/castor/cern.ch/cms","")'`
  echo "cmsStageIn $MSSCAFDIR/milleBinaryISN.dat milleBinaryISN.dat"
  cmsStageIn $MSSCAFDIR/milleBinaryISN.dat milleBinaryISN.dat
fi

# set up the CMS environment
cd $HOME/cms/CMSSW/CMSSW_3_0_0
eval `scramv1 runtime -sh`
rehash

cd $BATCH_DIR
echo Running directory changed to $(pwd).

# create link for treeFile(s) in mille job $RUNDIR's
# (comment in case your cfg is not creating treeFiles...)
ln -s $RUNDIR/../jobISN/treeFile.root treeFileISN.root

# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

# clean the link created above to avoid copying later (maybe uncomment, see above)
rm  treeFileISN.root

# Gzip one by one in case one argument cannot be expanded:
gzip -f *.log
gzip -f *.txt

# Merge possible alignment monitor and millepede monitor hists...
# ...and remove individual histogram files after merging to save space (if success):
# NOTE: the names "histograms.root" and "millePedeMonitor.root" must match what is in
#      your  alignment_cfg.py!
#hadd histograms_merge.root $RUNDIR/../job???/histograms.root
#if [ $? -eq 0 ]; then
#    rm $RUNDIR/../job???/histograms.root
#fi
hadd millePedeMonitor_merge.root $RUNDIR/../job???/millePedeMonitor.root
if [ $? -eq 0 ]; then
    rm $RUNDIR/../job???/millePedeMonitor.root
fi
# Macro creating millepede.his.ps with pede information hists:
if [ -e $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C ] ; then
    # Checked out version if existing:
    cp $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C .
else
    # If nothing checked out, take from release:
    cp $CMSSW_RELEASE_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C .
fi
root -b -q "readPedeHists.C+(\"print nodraw\")" 
gzip -f *.ps
echo "\nDirectory content after running cmsRun, zipping log file and merging histogram files:"
ls -lh
# Copy everything you need to MPS directory of your job,
# but you might want to copy less stuff to save disk space:
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.dump $RUNDIR
cp -p *.log.gz $RUNDIR
cp -p *.txt.gz $RUNDIR
cp -p *.ps.gz $RUNDIR
cp -p *.root $RUNDIR
cp -p millepede.*s $RUNDIR
cp -p *.db $RUNDIR
