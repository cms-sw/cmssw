#!/bin/zsh
#
# Run script template for Pede job, copying binary files from mass storage to local disk.
#
# Adjustments might be needed for CMSSW environment.

#temporary fix (?):
#unset PYTHONHOME

EOS="/afs/cern.ch/project/eos/installation/cms/bin/eos.select"
EOSPREFIX="root://eoscms//eos/cms"

cd $CMSSW_BASE/src
eval `scramv1 runtime -sh`
cd -

# these defaults will be overwritten by MPS
RUNDIR=$HOME/scratch0/some/path
MSSDIR=/castor/cern.ch/user/u/username/another/path
MSSDIRPOOL=
CONFIG_FILE=

export X509_USER_PROXY=${RUNDIR}/.user_proxy

#get list of treefiles
TREEFILELIST=
if [ "$MSSDIRPOOL" != "cmscafuser" ]; then
else
    TREEFILELIST=`${EOS} ls -l $MSSDIR | grep -i treeFile | grep -i root`
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
  MSSCAFDIR=`echo $MSSDIR | perl -pe 's/\/castor\/cern.ch\/cms//gi'`

  untilSuccess xrdcp ${EOSPREFIX}${MSSCAFDIR}/milleBinaryISN.dat.gz milleBinaryISN.dat.gz
  copytreefile xrdcp ${EOSPREFIX}${MSSCAFDIR}/treeFileISN.root treeFileISN.root
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
time cmsRun $CONFIG_FILE

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

# Macro creating chi2ndfperbinary.pdf with pede chi2/ndf information hists:
if [ -e $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C ] ; then
    # Checked out version if existing:
    cp $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C .
else
    # If nothing checked out, take from release:
    cp $CMSSW_RELEASE_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/createChi2ndfplot.C .
fi
mps_parse_pedechi2hist.py -d $RUNDIR/../../mps.db --his millepede.his -c $CONFIG_FILE
if [ -f chi2pedehis.txt ]; then
    root -l -x -b -q 'createChi2ndfplot.C+("chi2pedehis.txt")'
fi

# Macro creating millepede.his.pdf with pede information hists:
if [ -e $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C ] ; then
    # Checked out version if existing:
    cp $CMSSW_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C .
else
    # If nothing checked out, take from release:
    cp $CMSSW_RELEASE_BASE/src/Alignment/MillePedeAlignmentAlgorithm/macros/readPedeHists.C .
fi
root -b -q "readPedeHists.C+(\"print nodraw\")"

# zip plot files:
gzip -f *.pdf
# now zip .his and .res:
gzip -f millepede.*s
# in case of diagonalisation zip this:
gzip -f millepede.eve
# zip monitoring file:
gzip -f millepede.mon

#list IOVs
for tag in $(sqlite3 alignments_MP.db  "SELECT NAME FROM TAG;")
do
    conddb --db alignments_MP.db list ${tag}
done

#split the IOVs
aligncond_split_iov.sh alignments_MP.db alignments_split_MP.db

echo "\nDirectory content after running cmsRun, zipping log file and merging histogram files:"
ls -lh
# Copy everything you need to MPS directory of your job
# (separate cp's for each item, otherwise you loose all if one file is missing):
cp -p *.root $RUNDIR
cp -p *.gz $RUNDIR
cp -p *.db $RUNDIR
cp -p *.end $RUNDIR

# copy aligment_merge.py for mps_validate.py
cp -p $CONFIG_FILE alignment_merge.py
# run mps_validate.py
campaign=`basename $MSSDIR`
mps_validate.py -m $campaign -p ./

cp -pr validation_output $RUNDIR
