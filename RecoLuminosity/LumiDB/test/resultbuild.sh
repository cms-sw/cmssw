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

cd $workdir
eval `scramv1 runtime -sh`
export TNS_ADMIN=/home/lumidb
cd $macrodir
date > "$lumilogpath/resultbuild_tmp.log"
python $macrodir/lumibylsrecorder.py -b 209151 -o $outdir -i $indir -P /home/lumidb/auth/writer -s oracle://cms_orcon_prod/cms_lumi_prod -d oracle://cms_orcon_prod/cms_lumi_prod >> "$lumilogpath/resultbuild_tmp.log"
date >> "$lumilogpath/resultbuild_tmp.log"
mv "$lumilogpath/resultbuild_tmp.log" "$lumilogpath/resultbuild.log"
cd $pwd 

