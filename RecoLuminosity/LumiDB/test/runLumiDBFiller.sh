# ~cmssw2/scritps/setup.sh
source /nfshome0/cmssw/setup/group_aliases.sh;
export VO_CMS_SW_DIR="/nfshome0/cmssw2";
source /nfshome0/cmssw2/cmsset_default.sh;
export PATH="/nfshome0/cmssw2/scripts:${PATH}";

cd /nfshome0/hcallumipro/LumiDBUtil/exec/CMSSW_3_6_1/src/
eval `scramv1 runtime -sh`
export TNS_ADMIN=/home/lumidb

dbConnectionString="oracle://cms_orcon_prod/cms_lumi_prod"
dropboxDir="/dropbox/hcallumipro/"
normalization=6330
lumiauthpath="/home/lumidb/auth/writer"
lumilogpath="/home/lumidb/log"
date > "$lumilogpath/tmp.log"
lumiDBFiller.py -c $dbConnectionString -d $dropboxDir -norm $normalization -P $lumiauthpath -L $lumilogpath > "$lumilogpath/tmp.log"
date > "$lumilogpath/tmp.log"
myDate=`date +"%y-%m-%d-%H"`
mv "$lumilogpath/tmp.log" "$lumilogpath/lumiDBFiller-$myDate.log"
