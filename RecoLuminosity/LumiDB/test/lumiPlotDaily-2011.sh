#!/bin/sh
currendir=`pwd`
workdir="/build1/zx/cron/CMSSW_3_11_0"
authdir="/afs/cern.ch/user/x/xiezhen"
overviewdir="/afs/cern.ch/cms/lumi/www/plots/overview"
operationdir="/afs/cern.ch/cms/lumi/www/plots/operation"
physicsdir="/afs/cern.ch/cms/lumi/www/plots/physicscertified"
publicresultdir="/afs/cern.ch/cms/lumi/www/publicplots"
logpath="/afs/cern.ch/cms/lumi"

logname="lumiPlotDaily-2011.log"
logfilename="$logpath/$logname"
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
physicsselectionFile="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions11/7TeV/StreamExpress/goodrunlist_json.txt"
beamenergy="3500"
beamfluctuation="0.2"
beamstatus="stable"

source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch $logfilename
date >> $logfilename
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $overviewdir -beamstatus $beamstatus -beamenergy $beamenergy -beamfluctuation $beamfluctuation --withTextOutput total2011vstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $publicresultdir -beamstatus $beamstatus -beamenergy $beamenergy -beamfluctuation $beamfluctuation --withTextOutput total2011vstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $overviewdir -beamstatus $beamstatus -beamenergy $beamenergy -beamfluctuation $beamfluctuation --withTextOutput perday2011 >> $logfilename 
sleep 1;
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $publicresultdir -beamstatus $beamstatus -beamenergy $beamenergy -beamfluctuation $beamfluctuation --withTextOutput perday2011 >> $logfilename 
sleep 1;
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir --withTextOutput instpeak2011vstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $publicresultdir --withTextOutput instpeak2011vstime >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir -beamstatus $beamstatus  -beamenergy $beamenergy -beamfluctuation $beamfluctuation --withTextOutput total2011vsfill >> $logfilename 
sleep 1
lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $operationdir -beamstatus $beamstatus  -beamenergy $beamenergy -beamfluctuation $beamfluctuation --withTextOutput total2011vsrun >> $logfilename 
#sleep 1
#lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $physicsdir -i $physicsselectionFile --withTextOutput physicsvstime >> $logfilename
#sleep 1
#lumiPlotFiller.py -c $dbConnectionString -P $authdir -o $physicsdir -i $physicsselectionFile --withTextOutput physicsperday >> $logfilename
cd $currentdir
