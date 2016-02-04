#!/bin/sh
currendir=`pwd`
workdir="/build1/zx/cron/CMSSW_3_7_0_pre3"
#workdir="/afs/cern.ch/user/x/xiezhen/w1/lumidist/CMSSW_3_7_0_pre3"
authdir="/afs/cern.ch/user/x/xiezhen"
overviewdir="/afs/cern.ch/cms/lumi/www/plots/overview"
operationdir="/afs/cern.ch/cms/lumi/www/plots/operation"
physicsdir="/afs/cern.ch/cms/lumi/www/plots/physicscertified"
publicresultdir="/afs/cern.ch/cms/lumi/www/plots/publicresult"
logpath="/afs/cern.ch/cms/lumi"
logname="lumiPlotDaily.log"
logfilename="$logpath/$logname"
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
physicsselectionFile="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions10/7TeV/StreamExpress/goodrunlist_json.txt"
beamenergy="3.5e03"
beamstatus="stable"
beamfluctuation="0.1"

source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch $logfilename
date >> $logfilename
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $overviewdir -beamstatus $beamstatus --withTextOutput --annotateboundary totalvstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $publicresultdir -beamstatus $beamstatus --withTextOutput totalvstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $overviewdir -beamstatus $beamstatus --withTextOutput --annotateboundary perday >> $logfilename 
sleep 1;
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $publicresultdir -beamstatus $beamstatus --withTextOutput perday >> $logfilename 
sleep 1;
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir --withTextOutput --annotateboundary instpeakvstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $publicresultdir --withTextOutput instpeakvstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir -beamstatus $beamstatus --withTextOutput totalvsfill >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir -beamstatus $beamstatus --withTextOutput totalvsrun >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $physicsdir -i $physicsselectionFile --withTextOutput physicsvstime >> $logfilename
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $physicsdir -i $physicsselectionFile --withTextOutput physicsperday >> $logfilename
cd $currentdir
