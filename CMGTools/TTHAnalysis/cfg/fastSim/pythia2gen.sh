#!/bin/bash
TMPDIR=$PWD
#### ENV
cd /afs/cern.ch/user/g/gpetrucc/w/GENS/CMSSW_5_3_5/src
export SCRAM_ARCH=slc5_amd64_gcc462
eval $(scramv1 runtime -sh)
cd /afs/cern.ch/user/g/gpetrucc/ttH/CMGTools/CMSSW_5_3_5/src/CMGTools/TTHAnalysis/cfg/fastSim
#### CREATE CFG
M="default"; if [[ "$1" == "--D6T" || "$1" == "--Z2" ||  "$1" == "--Z2Star" ||  "$1" == "--ProQ20"  ||  "$1" == "--P11" ]]; then M="$1"; shift; fi;
OUTFILE=$1
OUTBASE=$(basename $OUTFILE .root)
echo "Will  write to $OUTFILE";
shift;

INCFG=$1
echo "Will generate using $INCFG"
shift;

LUMI=$1
echo "Will set lumisection number to $LUMI"
shift

cat $INCFG > jobs/$OUTBASE.cfg.py
echo "process.AODSIMoutput.fileName = '$TMPDIR/$OUTBASE.root'"   >> jobs/$OUTBASE.cfg.py
echo "process.source.firstLuminosityBlock = cms.untracked.uint32($LUMI)"   >> jobs/$OUTBASE.cfg.py

if [[ "$M" == "--D6T" ]]; then
    echo "Using D6T pythia tune"
    sed -i 's/PythiaUEZ2starSettings_cfi/PythiaUED6TSettings_cfi/' jobs/$OUTBASE.cfg.py
elif [[ "$M" == "--Z2" ]]; then
    echo "Using Z2 pythia tune"
    sed -i 's/PythiaUEZ2starSettings_cfi/PythiaUEZ2Settings_cfi/' jobs/$OUTBASE.cfg.py
elif [[ "$M" == "--Z2Star" ]]; then
    echo "Using Z2Star pythia tune (default)"
elif [[ "$M" == "--ProQ20" ]]; then
    echo "Using ProQ20 pythia tune"
    sed -i 's/PythiaUEZ2starSettings_cfi/PythiaUEProQ20Settings_cfi/' jobs/$OUTBASE.cfg.py
elif [[ "$M" == "--P11" ]]; then
    echo "Using P11 pythia tune"
    sed -i 's/PythiaUEZ2starSettings_cfi/PythiaUEP11Settings_cfi/' jobs/$OUTBASE.cfg.py
fi
cat >> jobs/$OUTBASE.cfg.py <<_EOF_
## Scramble
import random
rnd = random.SystemRandom()
for X in process.RandomNumberGeneratorService.parameterNames_(): 
   if X != 'saveFileName': getattr(process.RandomNumberGeneratorService,X).initialSeed = rnd.randint(1,99999999)
_EOF_
cmsRun jobs/$OUTBASE.cfg.py 2>&1 | tee jobs/$OUTBASE.log
test -f $TMPDIR/$OUTBASE.root && cmsStageIn $TMPDIR/$OUTBASE.root $OUTFILE
