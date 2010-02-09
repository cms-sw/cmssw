#!/bin/zsh 
#
# Run script template for Pede job, copying binary files from mass storage to local disk.
#
# Adjustments might be needed for CMSSW environment.

#temporary fix (?):
#unset PYTHONHOME

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
MSSDIRPOOL=

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

# AP 09.02.2010 - Create the fifo(s)
UUID=`uuidgen -r`
# rm -f /tmp/milleBinaryISN-$UUID.dat.gz
mkfifo /tmp/milleBinaryISN-$UUID.dat.gz

# stage and copy the binary file(s), first set castor pool for binary files in $MSSDIR area
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
# Not using cmscafuser pool => rfcp command must be used
  export STAGE_SVCCLASS=$MSSDIRPOOL
  stager_get -M $MSSDIR/milleBinaryISN.dat.gz
# AP 26.01.2010 - rfcp and gunzip at the same time, using the fifo(s)
  echo "rfcp $MSSCAFDIR/milleBinaryISN.dat.gz /tmp/milleBinaryISN-$UUID.dat.gz"
  rfcp $MSSDIR/milleBinaryISN.dat.gz /tmp/milleBinaryISN-$UUID.dat.gz &; cat /tmp/milleBinaryISN-$UUID.dat.gz | gzip -d -c > milleBinaryISN.dat
  stager_get -M $MSSDIR/treeFileISN.root
  rfcp $MSSDIR/treeFileISN.root $BATCH_DIR
else
# Using cmscafuser pool => cmsStageIn command must be used
  . /afs/cern.ch/cms/caf/setup.sh
  MSSCAFDIR=`echo $MSSDIR | awk 'sub("/castor/cern.ch/cms","")'`
  echo "cmsStageIn $MSSCAFDIR/milleBinaryISN.dat.gz /tmp/milleBinaryISN-$UUID.dat.gz"
# AP 26.01.2010 - rfcp and gunzip at the same time, using the fifo(s)
  cmsStageIn $MSSCAFDIR/milleBinaryISN.dat.gz /tmp/milleBinaryISN-$UUID.dat.gz &; cat /tmp/milleBinaryISN-$UUID.dat.gz | gzip -d -c > milleBinaryISN.dat
  echo "cmsStageIn $MSSCAFDIR/treeFileISN.root treeFileISN.root"
  cmsStageIn $MSSCAFDIR/treeFileISN.root treeFileISN.root
fi

# AP 21.01.2010 - remove the fifo(s)
rm -f /tmp/milleBinaryISN-$UUID.dat.gz

# set up the CMS environment
cd $HOME/cms/CMSSW/CMSSW_3_4_0
eval `scram runtime -sh`
rehash

cd $BATCH_DIR
echo Running directory changed to $(pwd).

# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

# clean up what has been staged in (to avoid copy mistakes...)
rm  treeFileISN.root

# Gzip one by one in case one argument cannot be expanded:
gzip -f *.log
gzip -f *.txt
gzip -f *.dump

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

# now zip .his and .res:
gzip -f millepede.*s
# in case of diagonalisation zip this:
gzip -f millepede.eve

echo "\nDirectory content after running cmsRun, zipping log file and merging histogram files:"
ls -lh
# Copy everything you need to MPS directory of your job
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.root $RUNDIR
cp -p *.gz $RUNDIR
cp -p *.db $RUNDIR
