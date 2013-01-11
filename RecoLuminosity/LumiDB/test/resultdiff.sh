#!/bin/sh
source /nfshome0/cmssw/setup/group_aliases.sh;
export VO_CMS_SW_DIR="/nfshome0/cmssw2";
export SCRAM_ARCH=slc5_amd64_gcc434
source /nfshome0/cmssw2/cmsset_default.sh;
export PATH="/nfshome0/cmssw2/scripts:${PATH}";

workdir="/nfshome0/hcallumipro/LumiDBUtil/exec/CMSSW_4_2_3/"
cd $workdir
eval `scramv1 runtime -sh`
export TNS_ADMIN=/home/lumidb

macrodir="$workdir/src/RecoLuminosity/LumiDB/test"
outdir="/home/lumidb/lumibylsresult/2012"
pwd=`pwd`
cd $workdir
eval `scramv1 runtime -sh`
cd $outdir
python $macrodir/lumibylsdiff.py > lumibylsdiff.out
cd $pwd 

