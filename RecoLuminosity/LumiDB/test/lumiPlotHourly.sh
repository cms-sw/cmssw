#!/bin/sh
currendir=`pwd`
workdir="/build1/zx/cron/CMSSW_3_7_0_pre3"
authdir="/afs/cern.ch/user/x/xiezhen"
outdir="/afs/cern.ch/cms/lumi/www/plots/operation"
logpath="/afs/cern.ch/cms/lumi/"
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch "$logpath/lumiPlotHourly.log"
date >> "$logpath/lumiPlotHourly.log"
cp  "$outdir/runlist.txt" "$outdir/runlist.txt.old"
lumiPlotFiller.py -c $dbConnectionString -P $authdir createrunlist -o $outdir >> "$logpath/lumiPlotHourly.log" ;
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir instperrun -o $outdir -i "$outdir/runlist.txt">> "$logpath/lumiPlotHourly.log"
#if `diff "$outdir/runlist.txt" "$outdir/runlist.txt.old" > /dev/null` ; then
#   echo 'no new runs, do nothing' >> "$logpath/lumiPlotHourly.log"
#else
#fi   
date >> "$logpath/lumiPlotHourly.log"
cd $currentdir
