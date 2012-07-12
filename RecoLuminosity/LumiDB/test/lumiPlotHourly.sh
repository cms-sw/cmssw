#!/bin/sh
currendir=`pwd`
sarch="slc5_amd64_gcc434"
export SCRAM_ARCH="$sarch"
workdir="/afs/cern.ch/user/l/lumipro/scratch0/exec/CMSSW_5_0_1"
authdir="/afs/cern.ch/user/l/lumipro"
outdir="/afs/cern.ch/cms/lumi/www/publicplots"
logpath="/afs/cern.ch/cms/lumi"
logfile="lumiPlot-2012pp.log"
beamenergy=4000
dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
amodetag="PROTPHYS"
begTime='04/04/12 00:00:00'
normStr="pp8TeV"
source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`

touch "$logpath/$logfile"

date >> "$logpath/$logfile"

outfile="$outdir/totallumivstime-pp-2012"
infile="$outdir/totallumivstime-pp-2012.csv"

touch $infile
echo "lumiPlot.py -c $dbConnectionString -P $authdir --norm $normStr -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin $begTime --inplotdata $infile --outplotdata $outfile  --lastpointfromdb time" >> "$logpath/$logfile"

lumiPlot.py -c $dbConnectionString -P $authdir  --norm $normStr -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin "$begTime" --inplotdata "$infile" --outplotdata "$outfile" --lastpointfromdb time >> "$logpath/$logfile"

date >> "$logpath/$logfile"

outfile="$outdir/lumiperday-pp-2012"
infile="$outdir/lumiperday-pp-2012.csv"
touch $infile

echo "lumiPlot.py -c $dbConnectionString -P $authdir --norm $normStr -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin $begTime --inplotdata $infile --outplotdata $outfile --lastpointfromdb perday">>  "$logpath/$logfile"

lumiPlot.py -c $dbConnectionString -P $authdir --norm $normStr -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin "$begTime" --inplotdata $infile --outplotdata $outfile --lastpointfromdb perday >> "$logpath/$logfile"

date >> "$logpath/$logfile"

outfile="$outdir/lumipeak-pp-2012"
infile="$outdir/lumipeak-pp-2012.csv"

touch $infile

echo "lumiPlot.py -c $dbConnectionString -P $authdir --norm $normStr -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin $begTime --inplotdata $infile --outplotdata $outfile --lastpointfromdb instpeakperday">> "$logpath/$logfile"

lumiPlot.py -c $dbConnectionString -P $authdir --norm $normStr -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin "$begTime" --inplotdata $infile --outplotdata $outfile --lastpointfromdb instpeakperday >> "$logpath/$logfile"

date >> "$logpath/$logfile"
cd $currentdir
