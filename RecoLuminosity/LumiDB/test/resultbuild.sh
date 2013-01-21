#!/bin/sh
source /nfshome0/cmssw/setup/group_aliases.sh;
export VO_CMS_SW_DIR="/nfshome0/cmssw2";
export SCRAM_ARCH=slc5_amd64_gcc434
source /nfshome0/cmssw2/cmsset_default.sh;
export PATH="/nfshome0/cmssw2/scripts:${PATH}";
workdir="/nfshome0/hcallumipro/LumiDBUtil/exec/CMSSW_4_2_3/"
lumilogpath="/home/lumidb/log"
pwd=`pwd`
macrodir="$workdir/src/RecoLuminosity/LumiDB/test"
outdir="/home/lumidb/lumibylsresult/2013"
indir="/home/lumidb/lumibylsresult/2012"
minbiasX=2100000.0
authpath="/home/lumidb/auth/writer"
sourcedb="oracle://cms_orcon_prod/cms_lumi_prod"
destdb="oracle://cms_orcon_prod/cms_lumi_prod"
minrun=209151

cd $workdir
eval `scramv1 runtime -sh`
export TNS_ADMIN=/home/lumidb

date > "$lumilogpath/resultbuild_tmp.log"
python $macrodir/lumibylsrecorder.py -b $minrun -o $outdir -i $indir -P $authpath -s $sourcedb -d $destdb --minBiasXsec $minbiasX>> "$lumilogpath/resultbuild_tmp.log"
date >> "$lumilogpath/resultbuild_tmp.log"
mv "$lumilogpath/resultbuild_tmp.log" "$lumilogpath/resultbuild.log"
cd $pwd 

