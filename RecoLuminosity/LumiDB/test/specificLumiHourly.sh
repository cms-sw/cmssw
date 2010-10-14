#!/bin/sh
currendir=`pwd`
workdir="/build1/zx/cron/CMSSW_3_7_0_pre3"
authdir="/afs/cern.ch/user/x/xiezhen"
indir="/afs/cern.ch/cms/lumi/lhcfills"
outdir="/afs/cern.ch/cms/lumi/specificlumi"
logpath="/afs/cern.ch/cms/lumi/"
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch "$logpath/specificLumiHourly.log"
date >> "$logpath/specificLumiHourly.log"
specificLumi.py -c $dbConnectionString -P $authdir -i $indir -o $outdir >> "$logpath/specificLumiHourly.log" ;
date >> "$logpath/specificLumiHourly.log"
cd $currentdir
