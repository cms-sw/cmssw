#!/bin/zsh 
#
# Run script template for Pede job, copying binary files from mass storage to local disk.
#
# Adjustments might be needed for CMSSW environment.

#temporary fix (?):
#unset PYTHONHOME

cd $CMSSW_BASE/src
eval `scramv1 runtime -sh`
cd -

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
MSSDIRPOOL=

#get list of treefiles
TREEFILELIST=
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
else
    TREEFILELIST=`cmsLs -l $MSSDIR | grep -i treeFile | grep -i root`
fi
if [ -z "$TREEFILELIST" ]; then
    echo "\nThe list of treefiles seems to be empty.\n"
fi

clean_up () {
#try to recover log files and root files
    echo try to recover log files and root files ...
    cp -p pede.dump* $RUNDIR
    cp -p *.txt.* $RUNDIR
    cp -p *.log $RUNDIR
    cp -p *.log.gz $RUNDIR
    cp -p millePedeMonitor*root $RUNDIR
    cp -p millepede.res* $RUNDIR
    cp -p millepede.end $RUNDIR
    cp -p millepede.his* $RUNDIR
    cp -p *.db $RUNDIR
    exit
}
#LSF signals according to http://batch.web.cern.ch/batch/lsf-return-codes.html
trap clean_up HUP INT TERM SEGV USR2 XCPU XFSZ IO


# a helper function to repeatedly try failing copy commands
untilSuccess () {
# trying "$1 $2 $3 > /dev/null" until success,
# break after $4 tries (with three arguments do up to 5 tries).
    if  [ $# -lt 3 -o $# -gt 4 ]; then
	echo $0 needs 3 or 4 arguments
	return 1
    fi
    
    integer TRIES=0
    integer MAX_TRIES=5
    if [ $# = 4 ]; then MAX_TRIES=$4; fi
    
    $1 $2 $3 > /dev/null 
    while [ $? -ne 0 ] ; do # if not successfull, retry...
	if [ $TRIES -ge $MAX_TRIES ] ; then # ... but not until infinity!
            echo $0: Give up doing \"$1 $2 $3 \> /dev/null\".
            return 1
	fi
	TRIES=$TRIES+1
	echo $0: WARNING, problems with \"$1 $2 $3 \> /dev/null\", try again.
	sleep $[$TRIES*5] # for before each wait a litte longer...
	$1 $2 $3 > /dev/null 
    done

    echo successsfully executed \"$1 $2 $3 \> /dev/null\" 
    return 0
}

copytreefile () {
    CHECKFILE=`echo $TREEFILELIST | grep -i $2`
    if [ -z "$TREEFILELIST" ]; then
        untilSuccess $1 $2 $3
    else
        if [ -n "$CHECKFILE" ]; then
            untilSuccess $1 $2 $3
        fi
    fi
}

# The batch job directory (will vanish after job end):
BATCH_DIR=$(pwd)
echo "Running at $(date) \n        on $HOST \n        in directory $BATCH_DIR."

# stage and copy the binary file(s), first set castor pool for binary files in $MSSDIR area
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
# Not using cmscafuser pool => rfcp command must be used
  export STAGE_SVCCLASS=$MSSDIRPOOL
  export STAGER_TRACE=
  stager_get -M $MSSDIR/milleBinaryISN.dat.gz
  untilSuccess rfcp $MSSDIR/milleBinaryISN.dat.gz $BATCH_DIR
  stager_get -M $MSSDIR/treeFileISN.root
  copytreefile rfcp $MSSDIR/treeFileISN.root $BATCH_DIR
else
# Using cmscafuser pool => cmsStageIn command must be used
  . /afs/cern.ch/cms/caf/setup.sh
  #MSSCAFDIR=`echo $MSSDIR | awk 'sub("/castor/cern.ch/cms","")'`
  MSSCAFDIR=`echo $MSSDIR | perl -pe 's/\/castor\/cern.ch\/cms//gi'`
  
  untilSuccess cmsStageIn $MSSCAFDIR/milleBinaryISN.dat.gz milleBinaryISN.dat.gz
  copytreefile cmsStageIn $MSSCAFDIR/treeFileISN.root treeFileISN.root
fi

# We have gzipped binaries, but the python config looks for .dat
# (could also try to substitute in config ".dat" with ".dat.gz"
#  ONLY for lines which contain "milleBinary" using "sed '/milleBinary/s/.dat/.dat.gz/g'"):
ln -s milleBinaryISN.dat.gz milleBinaryISN.dat

# set up the CMS environment
cd CMSSW_RELEASE_AREA
eval `scram runtime -sh`
rehash

cd $BATCH_DIR
echo Running directory changed to $(pwd).

echo "\nDirectory content before running cmsRun:"
ls -lh
# Execute. The cfg file name will be overwritten by MPS
time cmsRun the.cfg

# clean up what has been staged in (to avoid copy mistakes...)
rm treeFileISN.root
rm milleBinaryISN.dat.gz milleBinaryISN.dat

# Gzip one by one in case one argument cannot be expanded:
gzip -f *.log
gzip -f *.txt
gzip -f *.dump

#Try to merge millepede monitor files. This only works successfully if names were assigned to jobs.
mps_merge_millepedemonitor.pl $RUNDIR/../../mps.db $RUNDIR/../../

# Merge possible alignment monitor and millepede monitor hists...
# ...and remove individual histogram files after merging to save space (if success):
# NOTE: the names "histograms.root" and "millePedeMonitor.root" must match what is in
#      your  alignment_cfg.py!
#hadd histograms_merge.root $RUNDIR/../job???/histograms.root
#if [ $? -eq 0 ]; then
#    rm $RUNDIR/../job???/histograms.root
#fi
hadd millePedeMonitor_merge.root $RUNDIR/../job???/millePedeMonitor*.root
if [ $? -eq 0 ]; then
    rm $RUNDIR/../job???/millePedeMonitor*.root
fi

# Macro creating chi2ndfperbinary.eps with pede chi2/ndf information hists:
if [ -e $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C ] ; then
    # Checked out version if existing:
    cp $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C .
else
    # If nothing checked out, take from release:
    cp $CMSSW_RELEASE_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C .
fi
mps_parse_pedechi2hist.pl $RUNDIR/../../mps.db millepede.his the.cfg
if [ -f chi2pedehis.txt ]; then
    root -l -x -b -q 'createChi2ndfplot.C+("chi2pedehis.txt")'
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

#list IOVs
cmscond_list_iov -c sqlite_file:alignments_MP.db -t Alignments
cmscond_list_iov -c sqlite_file:alignments_MP.db -t Deformations
cmscond_list_iov -c sqlite_file:alignments_MP.db -t SiStripLorentzAngle_deco
cmscond_list_iov -c sqlite_file:alignments_MP.db -t SiStripLorentzAngle_peak
cmscond_list_iov -c sqlite_file:alignments_MP.db -t SiPixelLorentzAngle
cmscond_list_iov -c sqlite_file:alignments_MP.db -t SiStripBackPlaneCorrection

#split the IOVs
aligncond_split_iov -s sqlite_file:alignments_MP.db -i Alignments -d sqlite_file:alignments_split_MP.db -t Alignments
aligncond_split_iov -s sqlite_file:alignments_MP.db -i Deformations -d sqlite_file:alignments_split_MP.db -t Deformations
aligncond_split_iov -s sqlite_file:alignments_MP.db -i SiStripLorentzAngle_deco -d sqlite_file:alignments_split_MP.db -t SiStripLorentzAngle_deco
aligncond_split_iov -s sqlite_file:alignments_MP.db -i SiStripLorentzAngle_peak -d sqlite_file:alignments_split_MP.db -t SiStripLorentzAngle_peak
aligncond_split_iov -s sqlite_file:alignments_MP.db -i SiPixelLorentzAngle -d sqlite_file:alignments_split_MP.db -t SiPixelLorentzAngle
aligncond_split_iov -s sqlite_file:alignments_MP.db -i SiStripBackPlaneCorrection -d sqlite_file:alignments_split_MP.db -t SiStripBackPlaneCorrection

echo "\nDirectory content after running cmsRun, zipping log file and merging histogram files:"
ls -lh
# Copy everything you need to MPS directory of your job
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.root $RUNDIR
cp -p *.gz $RUNDIR
cp -p *.db $RUNDIR
cp -p *.end $RUNDIR

if [ -f chi2ndfperbinary.eps ]; then
    gzip -f chi2ndfperbinary.eps
    cp -p chi2ndfperbinary.eps.gz $RUNDIR
fi

if [ -f chi2ndfperbinary.C ]; then
    gzip -f chi2ndfperbinary.C
    cp -p chi2ndfperbinary.C.gz $RUNDIR
fi
