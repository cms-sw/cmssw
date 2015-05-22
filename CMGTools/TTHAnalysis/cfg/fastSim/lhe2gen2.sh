#!/bin/bash
TMPDIR=$PWD
#### ENV
cd /afs/cern.ch/user/g/gpetrucc/w/GENS/CMSSW_5_3_5/src
export SCRAM_ARCH=slc5_amd64_gcc462
eval $(scramv1 runtime -sh)
SRC=/afs/cern.ch/user/g/gpetrucc/ttH/CMGTools/CMSSW_5_3_5/src/CMGTools/TTHAnalysis/cfg/fastSim;
#### CREATE CFG
M="default"; if [[ "$1" == "--up" || "$1" == "--down" || "$1" == "--nojets" || "$1" == "--xqtup" || "$1" == "--xqtdown"  ]]; then M="$1"; shift; fi;
EVENTS=""
if [[ "$1" == "--events" ]]; then
    EVENTS="process.source.skipEvents = cms.untracked.uint32($2); process.maxEvents.input = cms.untracked.int32($3);"
    shift; shift; shift;
fi;
OUTFILE=$1
OUTBASE=$(basename $OUTFILE .root)
echo "Will  write to $OUTFILE";
shift;

INFILES=""; COUNTER=0
while [[ "$1" != "" ]]; do
    INFILE=$1; shift
    echo "Will read from $INFILE";
    if echo $INFILE | grep -q -v ^/store/; then
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

cat $SRC/Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff_py_noCMS_GEN.py > $OUTBASE.cfg.py
echo "process.source.fileNames = [ $INFILES ]"                   >> $OUTBASE.cfg.py
echo "process.AODSIMoutput.fileName = '$TMPDIR/$OUTBASE.root'"   >> $OUTBASE.cfg.py
NJETS=99
if   echo $OUTFILE | grep '_012jets' -q ; then NJETS=2;
elif echo $OUTFILE | grep '_01jets'  -q ; then NJETS=1;
elif [[ "$M" == "--nojets" ]]; then NJETS=0;
else echo "So il c...o io quanti jet vuoi. NJETS=$NJETS"; fi
echo "process.generator.jetMatching.MEMAIN_maxjets = $NJETS"   >> $OUTBASE.cfg.py
 
if [[ "$M" == "--up" ]]; then
    echo "Matching scale UP"
    echo "process.generator.PythiaParameters.processParameters += [ 'PARP(64)=4.', 'PARP(72)=0.125' ]"   >> $OUTBASE.cfg.py
elif [[ "$M" == "--down" ]]; then
    echo "Matching scale DOWN"
    echo "process.generator.PythiaParameters.processParameters += [ 'PARP(64)=0.25', 'PARP(72)=0.5' ]"   >> $OUTBASE.cfg.py
elif [[ "$M" == "--xqtup" ]]; then
    echo "Matching cut DOWN"
    echo "process.generator.jetMatching.MEMAIN_qcut = 20"   >> $OUTBASE.cfg.py
elif [[ "$M" == "--xqtdown" ]]; then
    echo "Matching cut DOWN"
    echo "process.generator.jetMatching.MEMAIN_qcut = 5"   >> $OUTBASE.cfg.py
elif [[ "$M" == "--nojets" ]]; then
    echo "No jet matching"
    echo "del process.generator.jetMatching"   >> $OUTBASE.cfg.py
fi
cat >> $OUTBASE.cfg.py <<_EOF_
## If needed, select events to process
$EVENTS
## Scramble
import random
rnd = random.SystemRandom()
for X in process.RandomNumberGeneratorService.parameterNames_(): 
   if X != 'saveFileName': getattr(process.RandomNumberGeneratorService,X).initialSeed = rnd.randint(1,99999999)
_EOF_
cmsRun $OUTBASE.cfg.py 2>&1 | tee $OUTBASE.log
#test -f $TMPDIR/$OUTBASE.root && cmsStageIn $TMPDIR/$OUTBASE.root $OUTFILE
#~/sh/skimreport $OUTBASE.log > $SRC/jobs/$OUTBASE.skimreport
#gzip $OUTBASE.log && cp -v $OUTBASE.log.gz $SRC/jobs/
