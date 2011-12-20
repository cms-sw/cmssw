#!/bin/sh

# This script reads online conditions from the specified run, writes them
# to a sqlite file (l1config.db), and assigns them an infinite IOV starting
# at run 1.

nflag=0
pflag=0
while getopts 'nph' OPTION
  do
  case $OPTION in
      n) nflag=1
          ;;
      p) pflag=1
	  ;;
      h) echo "Usage: [-n] runnum"
          echo "  -n: no RS"
          echo "  -p: centrally installed release, not on local machine"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

# get argument
runnum=$1

if [ $# -lt 1 ]
    then
    echo "Wrong number of arguments.  Usage: $0 runnum"
    exit 127
fi

if [ -f l1config.db ]
    then
    mv l1config.db l1config.db.save
fi

if [ ${pflag} -eq 0 ]
    then
    export SCRAM_ARCH=""
    export VO_CMS_SW_DIR=""
    source /opt/cmssw/cmsset_default.sh
else
    source /nfshome0/cmssw2/scripts/setup.sh
fi
eval `scramv1 run -sh`

echo "`date` : initializing sqlite file"
if [ -e $CMSSW_BASE/src/CondFormats/L1TObjects/xml ]
    then
    $CMSSW_BASE/src/CondTools/L1Trigger/test/bootstrap.com -l
else
    $CMSSW_RELEASE_BASE/src/CondTools/L1Trigger/test/bootstrap.com
fi

# copy default objects
cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadCondDB_cfg.py inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb runNumber=${runnum}
cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVDummy_cfg.py useO2OTags=1

exit ${o2ocode}
