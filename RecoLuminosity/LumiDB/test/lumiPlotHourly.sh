#!/bin/sh
workdir="/afs/cern.ch/user/l/lumipro/scratch0/exec/CMSSW_5_3_2"
authdir="/afs/cern.ch/user/l/lumipro"
macrodir="$workdir/src/RecoLuminosity/LumiDB/plotdata"
outdir="/afs/cern.ch/cms/lumi/www/publicplots"
logpath="/afs/cern.ch/cms/lumi"

if [[ $# -lt 1 ]]; then
  echo "ERROR Need at least a config file name in order to do something"
  exit 1
fi

if [[ $# -gt 2 ]]; then
  echo "ERROR Can only handle one config file name and an optional --ignore-cache flag"
  exit 1
fi

# Figure out the configuration file to use.
cfgfile=$macrodir/$1
logfile=${1/cfg/log}
if [[ ! -f $cfgfile ]]; then
  echo "ERROR File $cfgfile does not exist"
  exit 1
fi

# See if we are being asked to clean the cache.
IGNORE_CACHE_FLAG=""
if [[ $# -eq 2 ]]; then
  if [ "$2" != "--ignore-cache" ]; then
    echo "WARNING: Don't understand the '$2' flag --> ignoring it"
  else
    echo "WARNING: Ignoring and rebuilding the lumiCalc cache"
    IGNORE_CACHE_FLAG="--ignore-cache"
  fi
fi

source /afs/cern.ch/cms/cmsset_default.sh;
cd $workdir
eval `scramv1 runtime -sh`

touch $logpath/$logfile

date >> $logpath/$logfile

cd $outdir
python -u $macrodir/create_public_lumi_plots.py $cfgfile ${IGNORE_CACHE_FLAG} >> $logpath/$logfile

date >> $logpath/$logfile
