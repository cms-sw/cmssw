#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Usage: BitSynchro.sh [DatasSet] [RunNumber] <NumEvents> <BX>"
    echo "DataSet=MinBias|ZeroBias|ExpressPhysics|MinBiasRaw|ZeroBiasRaw"
    echo "<NumEvents> is optional, to limit number of processed events"
    echo "<BX> is optional, if set, look only around specified BX"
    exit
fi
if [ ! -n "$CAF_TRIGGER" ]; then
    echo "Problem! CAF_TRIGGER environment variable is not set."
    echo "Did you run the setup file?"
    echo "Now Exiting!"
    exit
fi
FILE="$CAF_TRIGGER/l1analysis/ntuples/""$1""_""$2"".root" 
if [ ! -e "$FILE" ]; then
    echo "Warning, file $FILE does not exist. Check dataset and run number."
    echo "If they are really what you want, you may need to produce the Ntuple before continuing."
    echo "Now Exiting!"
    exit
fi
if [ -d L1BitCorr_$1_$2 ]; then
    echo "Warning, expected output directory L1BitCorr_$1_$2 already exisiting."
    echo "Please delete this directory if you want to run this macro with dataset=$1 and run=$2."
    echo "Now Exiting!"
    exit
fi
NUMEV="-1"
if [ $# -gt 2 ]; then
    NUMEV=$3
fi
BX="-1"
if [ $# -gt 3 ]; then
    BX=$4
fi
echo "{gROOT->ProcessLine(\".x ""$CMSSW_BASE""/src/UserCode/L1TriggerDPG/macros/initL1Analysis.C\");gROOT->ProcessLine(\".L L1BitCorrV2.C+\");gStyle->SetPalette(1);L1BitCorr tool(\"$FILE\");tool.run(1040,46,$BX,$NUMEV,true);delete c1;tool.run(1040,15,$BX,$NUMEV,true);delete c1;tool.run(1040,54,$BX,$NUMEV,true);delete c1;tool.run(1040,55,$BX,$NUMEV,true);delete c1;tool.run(1040,1031,$BX,$NUMEV,true);delete c1;}" > tmp.C

root -l -q -b tmp.C
mkdir  L1BitCorr_$1_$2
mv L1BitCorr_*.gif L1BitCorr_$1_$2/.
rm tmp.C
echo "==="
echo "Done, plots saved in ./L1BitCorr_$1_$2/  :"
echo
ls ./L1BitCorr_$1_$2/
