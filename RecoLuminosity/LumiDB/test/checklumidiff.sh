#!/bin/sh
workdir="/afs/cern.ch/user/l/lumipro/scratch0/exec/CMSSW_5_3_2"
macrodir="$workdir/src/RecoLuminosity/LumiDB/plotdata"
outdir="/afs/cern.ch/cms/lumi/www"
pwd=`pwd`

source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
cd $outdir
python $macrodir/checklumidiff.py
cd $pwd 

