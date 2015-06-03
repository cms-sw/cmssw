#!/bin/bash
#
#  usage:   lhe2gen2.sh [ --events <skipEvents> <maxEvents> ] [ --firstLuminosityBlock <firstLumiBlock>] /store/path/output_file.root inputfile.lhe[.gz] [ inputfile2.lhe[.gz] ... ]
#
TMPDIR=$PWD
#### ENV
cd /afs/cern.ch/work/e/emanuele/monox/heppy/CMSSW_7_2_3_patch1/src/
export SCRAM_ARCH=slc6_amd64_gcc481
eval $(scramv1 runtime -sh)
SRC=/afs/cern.ch/work/e/emanuele/monox/heppy/CMSSW_7_2_3_patch1/src/CMGTools/MonoXAnalysis/prod/
#### CREATE CFG
SLHA=""
if [[ "$1" == "--slha" ]]; then
    SLHA=$2;
    echo "Will use SLHA $SLHA for the decays";
    shift; shift;
fi;

EVENTS=""
if [[ "$1" == "--events" ]]; then
    EVENTS="process.source.skipEvents = cms.untracked.uint32($2); process.maxEvents.input = cms.untracked.int32($3);"
    echo "Will process $3 events skipping the first $2"
    shift; shift; shift;
fi;
if [[ "$1" == "--firstLuminosityBlock" ]]; then
    LUMIS="process.source.firstLuminosityBlock = cms.untracked.uint32($2)"
    echo "Will set firstLuminosityBlock to $2"
    shift; shift;
fi;
CFGFILE=$1; shift;
echo " $SRC/$CFGFILE "
if [[ "$CFGFILE" == "" ]] || test \! -f $SRC/$CFGFILE; then
    echo "Missing cfg file $CFGFILE under $SRC"
fi;
OUTFILE=$1
OUTBASE=$(basename $OUTFILE .root)
echo "Will  write to $OUTFILE";
shift;

INFILES=""; COUNTER=0
while [[ "$1" != "" ]]; do
    INFILE=$1; shift
    echo "Will read from $INFILE";
    if [[ "$SLHA" != "" ]]; then
        ORIGINAL=$TMPDIR/events_original.$COUNTER.lhe
        CAT="cat"
        MODIFIED=$TMPDIR/events_slha.$COUNTER.lhe
        COUNTER=$(($COUNTER+1))
        if echo $INFILE | grep -q ^/store/; then
            echo "Getting $INFILE from EOS with cmsStageIn"
            cmsStageIn $INFILE $ORIGINAL
        elif echo $INFILE | grep -q ^http; then
            echo "Getting $INFILE from the web with wget"
            wget -O $ORIGINAL $INFILE 
        else
            echo "Input file $INFILE is already AFS acessible"
            ORIGINAL=$INFILE
        fi;
        if echo $INFILE | grep -q lhe.gz; then 
            CAT="zcat"; 
            echo "Input file is compressed"
        fi;
        echo "Running: $CAT $ORIGINAL | perl $SRC/replace_slha.pl $SLHA > $MODIFIED;"
        $CAT $ORIGINAL | perl $SRC/replace_slha.pl $SLHA > $MODIFIED;
        if [[ "$INFILES" != "" ]]; then
            INFILES="$INFILES, 'file:$MODIFIED'"; 
        else
            INFILES="'file:$MODIFIED'"; 
        fi
    elif echo $INFILE | grep -q -v ^/store/; then
        if echo $INFILE | grep -q lhe.gz; then  
            echo "Unzipping $INFILE in $TMPDIR/events.$COUNTER.lhe"
            zcat $INFILE > $TMPDIR/events.$COUNTER.lhe
            INFILE="file:$TMPDIR/events.$COUNTER.lhe";
            COUNTER=$(($COUNTER+1))
        else
            INFILE="file:$INFILE";
        fi
    fi
    if [[ "$INFILES" != "" ]]; then
        INFILES="$INFILES, '$INFILE'"; 
    else
        INFILES="'$INFILE'"; 
    fi
done

cd $TMPDIR;

cat $SRC/$CFGFILE > $OUTBASE.cfg.py
echo "process.source.fileNames = [ $INFILES ]"                   >> $OUTBASE.cfg.py
echo "process.output.fileName = '$TMPDIR/$OUTBASE.root'"   >> $OUTBASE.cfg.py

cat >> $OUTBASE.cfg.py <<_EOF_
## If needed, select events to process
$EVENTS
$LUMIS
## Scramble
import random
rnd = random.SystemRandom()
for X in process.RandomNumberGeneratorService.parameterNames_(): 
   if X != 'saveFileName': getattr(process.RandomNumberGeneratorService,X).initialSeed = rnd.randint(1,99999999)
_EOF_
cmsRun $OUTBASE.cfg.py 2>&1 | tee $OUTBASE.log
test -f $TMPDIR/$OUTBASE.root && cmsStageIn -f $TMPDIR/$OUTBASE.root $OUTFILE
~/sh/skimreport $OUTBASE.log > $SRC/jobs/$OUTBASE.skimreport
gzip $OUTBASE.log && cp -v $OUTBASE.log.gz $SRC/jobs/
if cmsLs $OUTFILE; then
    echo "Copied ok"
else
    test -f $TMPDIR/$OUTBASE.root && cmsStageIn -f $TMPDIR/$OUTBASE.root $OUTFILE 2>&1
fi;
