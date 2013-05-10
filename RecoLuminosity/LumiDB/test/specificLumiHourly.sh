#!/bin/sh
currendir=`pwd`
sarch="slc5_amd64_gcc434"
workdir="/afs/cern.ch/user/l/lumipro/scratch0/exec/CMSSW_5_0_1"
authdir="/afs/cern.ch/user/l/lumipro"
runfillmapdir="/afs/cern.ch/cms/lumi/ppfills2012"
outdir="/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/LHCFILES"
logpath="/afs/cern.ch/cms/lumi/"
logfile="specificLumi-2012pp.log"
minfill=2454

dbConnectionString="oracle://cms_orcoff_prod/cms_lumi_prod"
source /afs/cern.ch/cms/cmsset_default.sh;
export SCRAM_ARCH="$sarch";
cd $workdir
eval `scramv1 runtime -sh`

touch "$logpath/$logfile"

date >> "$logpath/$logfile"

echo "dumpFill.py -c $dbConnectionString -P $authdir -o $runfillmapdir --amodetag PROTPHYS --minfill $minfill" >> "$logpath/$logfile"

dumpFill.py -c $dbConnectionString -P $authdir -o $runfillmapdir --amodetag PROTPHYS --minfill $minfill

date >> "$logpath/$logfile"

echo "summaryLumi.py -c $dbConnectionString -P $authdir -i $runfillmapdir -o $outdir --amodetag PROTPHYS --minfill $minfill" >> "$logpath/$logfile"

summaryLumi.py -c $dbConnectionString -P $authdir -i $runfillmapdir -o $outdir --amodetag PROTPHYS --minfill $minfill >> "$logpath/$logfile" 

date >> "$logpath/$logfile"

echo "specificLumi.py -c $dbConnectionString -P $authdir -i $runfillmapdir -o $outdir --minfill $minfill" >> "$logpath/$logfile"

specificLumi.py -c $dbConnectionString -P $authdir -i $runfillmapdir -o $outdir --minfill $minfill >> "$logpath/$logfile"

date >> "$logpath/$logfile"

cd $currentdir
