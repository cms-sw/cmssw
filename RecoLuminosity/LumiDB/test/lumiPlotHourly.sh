#!/bin/sh
currendir=`pwd`
workdir="/afs/cern.ch/user/x/xiezhen/w1/luminewschema/CMSSW_4_2_3"
authdir="/afs/cern.ch/user/x/xiezhen"
outdir="/afs/cern.ch/user/x/xiezhen/w1/luminewschema/CMSSW_4_2_3"
logpath="."
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
amodetag="IONPHYS"
begTime='11/11/11 00:00:00'
normStr="hi7TeV"
source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`
touch "$logpath/lumiPlotCron.log"
date >> "$logpath/lumiPlotCron.log"

outfile="$outdir/totallumivstime-hi-2011"
infile="$outdir/totallumivstime-hi-2011.csv"
touch $infile
echo "lumiSumPlot.py -c $dbConnectionString -P $authdir -norm $normStr -b stable -beamenergy 3500 -beamfluctuation 0.2 -amodetag $amodetag -beginTime $begTime -inplot $outfile -outplot $outfile --without-correction time" >> "$logpath/lumiPlotCron.log"

lumiSumPlot.py -c $dbConnectionString -P $authdir  -norm $normStr -b stable -beamenergy 3500 -beamfluctuation 0.2 -amodetag $amodetag -beginTime "$begTime" -inplot "$infile" -outplot "$outfile"  --without-correction time >> "$logpath/lumiPlotCron.log"
echo DONE
outfile="$outdir/lumiperday-hi-2011"
infile="$outdir/lumiperday-hi-2011.csv"
touch $infile
echo "lumiSumPlot.py -c $dbConnectionString -P $authdir -norm $normStr -b stable -beamenergy 3500 -beamfluctuation 0.2 -amodetag $amodetag -beginTime $begTime -inplot $infile -outplot $outfile  perday ">>  "$logpath/lumiPlotCron.log"
echo 1
lumiSumPlot.py -c $dbConnectionString -P $authdir -norm $normStr -b stable -beamenergy 3500 -beamfluctuation 0.2 -amodetag $amodetag -beginTime "$begTime" -inplot $infile -outplot $outfile  --without-correction perday >> "$logpath/lumiPlotCron.log"
echo DONE
outfile="$outdir/lumipeak-hi-2011"
infile="$outdir/lumipeak-hi-2011.csv"
touch $infile
echo "lumiSumPlot.py -c $dbConnectionString -P $authdir -o $outdir -norm $normStr -b stable -beamenergy 3500 -beamfluctuation 0.2 -amodetag $amodetag -beginTime $begTime -inplot $infile -outplot $outfile instpeakperday">> "$logpath/lumiPlotCron.log"

lumiSumPlot.py -c $dbConnectionString -P $authdir -o $outdir -norm $normStr -b stable -beamenergy 3500 -beamfluctuation 0.2 -amodetag $amodetag -beginTime "$begTime" -inplot $infile -outplot $outfile  --without-correction instpeakperday >> "$logpath/lumiPlotCron.log"
echo DONE
date >> "$logpath/lumiPlotCron.log"
cd $currentdir
