#!/bin/sh
currendir=`pwd`
workdir="/build1/zx/cron/CMSSW_3_11_0"
authdir="/afs/cern.ch/user/x/xiezhen"
indir="/afs/cern.ch/cms/lumi/ppfills2011"
outdir="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/LHCFILES"
logpath="/afs/cern.ch/cms/lumi/"
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch "$logpath/specificLumiHourly-2011.log"
date >> "$logpath/specificLumiHourly-2011.log"
specificLumi-2011.py -c $dbConnectionString -P $authdir -i $indir -o $outdir >> "$logpath/specificLumiHourly-2011.log" ;
date >> "$logpath/specificLumiHourly-2011.log"
cd $currentdir
