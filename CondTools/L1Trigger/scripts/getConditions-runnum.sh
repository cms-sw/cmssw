#!/bin/sh

# This script reads online conditions from the specified run, writes them
# to a sqlite file (l1config.db), and assigns them an infinite IOV starting
# at run 1.

tagbase=CRAFT09

nflag=0
while getopts 'nh' OPTION
  do
  case $OPTION in
      n) nflag=1
          ;;
      h) echo "Usage: [-n] runnum"
          echo "  -n: no RS"
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

source /nfshome0/cmssw2/scripts/setup.sh
export SCRAM_ARCH=slc5_ia32_gcc434
eval `scramv1 run -sh`

echo "`date` : initializing sqlite file"
if [ -e $CMSSW_BASE/src/CondFormats/L1TObjects/xml ]
    then
    $CMSSW_BASE/src/CondTools/L1Trigger/test/bootstrap.com -l
else
    $CMSSW_RELEASE_BASE/src/CondTools/L1Trigger/test/bootstrap.com
fi

# copy default objects
cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadCondDB_cfg.py tagBase=${tagbase}_hlt inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb runNumber=${runnum}
cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVDummy_cfg.py tagBase=${tagbase}_hlt

exit ${o2ocode}
