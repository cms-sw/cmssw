#!/bin/sh
currendir=`pwd`
sarch="slc5_amd64_gcc462"
export SCRAM_ARCH="$sarch"
workdir="/afs/cern.ch/user/l/lumipro/scratch0/exec/CMSSW_5_3_2"
authdir="/afs/cern.ch/user/l/lumipro"
macrodir="$workdir/src/RecoLuminosity/LumiDB/plotdata"
outdir="/afs/cern.ch/cms/lumi/www/publicplots"
logpath="/afs/cern.ch/cms/lumi"
logfile="lumiPlot-2012pp.log"
beamenergy=4000
dbConnectionString="oracle://cms_orcon_adg/cms_lumi_prod"
amodetag="PROTPHYS"
begTime='04/04/12 00:00:00'
source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`

touch "$logpath/$logfile"

date >> "$logpath/$logfile"

outfile="$outdir/totallumivstime-pp-2012"
infile="$outdir/totallumivstime-pp-2012.csv"

export TNS_ADMIN=/afs/cern.ch/cms/lumi/DB
  
touch $infile
echo "lumiPlot.py -c $dbConnectionString -P $authdir -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin $begTime --outplotdata $outfile --inplotdata $infile --lastpointfromdb --without-png time" >> "$logpath/$logfile"

lumiPlot.py -c $dbConnectionString -P $authdir -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin "$begTime" --outplotdata "$outfile"  --inplotdata $infile --lastpointfromdb --without-png time >> "$logpath/$logfile"

date >> "$logpath/$logfile"

outfile="$outdir/lumiperday-pp-2012"
infile="$outdir/lumiperday-pp-2012.csv"
touch $infile

echo "lumiPlot.py -c $dbConnectionString -P $authdir -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin $begTime --outplotdata $outfile  --inplotdata $infile --lastpointfromdb --without-png perday">>  "$logpath/$logfile"

lumiPlot.py -c $dbConnectionString -P $authdir -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin "$begTime" --outplotdata $outfile  --inplotdata $infile --lastpointfromdb --without-png perday >> "$logpath/$logfile"

date >> "$logpath/$logfile"

outfile="$outdir/lumipeak-pp-2012"
infile="$outdir/lumipeak-pp-2012.csv"

touch $infile

echo "lumiPlot.py -c $dbConnectionString -P $authdir -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin $begTime --outplotdata $outfile  --inplotdata $infile --lastpointfromdb --without-png instpeakperday">> "$logpath/$logfile"

lumiPlot.py -c $dbConnectionString -P $authdir -b stable --beamenergy $beamenergy --beamfluctuation 0.15 --amodetag $amodetag --begin "$begTime" --outplotdata $outfile  --inplotdata $infile --lastpointfromdb --without-png instpeakperday >> "$logpath/$logfile"

cd $macrodir
echo "root -b -q create_public_lumi_plots.C">>"$logpath/$logfile"
root -b -q create_public_lumi_plots.C >> "$logpath/$logfile"
/bin/cp int_*.png $outdir
/bin/rm int_*.png
/bin/cp peak_lumi*.png $outdir
/bin/rm peak_lumi*.png
cd $currentdir
date >> "$logpath/$logfile"
