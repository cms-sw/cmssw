#!/bin/sh
currendir=`pwd`
workdir="/build1/zx/cron/CMSSW_3_7_0_pre3"
authdir="/afs/cern.ch/user/x/xiezhen"
operationdir="/afs/cern.ch/cms/lumi/www/plots/operation"
logpath="/afs/cern.ch/cms/lumi"
logname="lumiPlotWeekly.log"
logfilename="$logpath/$logname"
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"

source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch $logfilename
date >> $logfilename
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir totallumilastweek --withTextOutput >> $logfilename 
date >> $logfilename
cd $currentdir
