#!/bin/sh

tagbase=CRAFT09

nflag=0
while getopts 'nh' OPTION
  do
  case $OPTION in
      n) nflag=1
          ;;
      h) echo "Usage: [-n] tsckey"
          echo "  -n: no RS"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

# get argument
key=$1

if [ $# -lt 1 ]
    then
    echo "Wrong number of arguments.  Usage: $0 tsckey"
    exit 127
fi

if [ -f l1config.db ]
    then
    mv l1config.db l1config.db.save
fi

ln -sf /nfshome0/centraltspro/secure/authentication.xml .

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
cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadCondDB_cfg.py tagBase=${tagbase}_hlt inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb
cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVDummy_cfg.py tagBase=${tagbase}_hlt

echo "`date` : writing TSC payloads"
$CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-key.sh -c ${key} ${tagbase}
o2ocode=$?

if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: TSC payloads OK"
else
    echo "L1-O2O-ERROR: TSC payloads not OK!" >&2
    exit ${o2ocode}
fi

echo "`date` : setting TSC IOVs"
$CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-iov.sh 10 ${key} ${tagbase}
o2ocode=$?

if [ ${o2ocode} -eq 0 ]
    then
    echo "L1-O2O-INFO: TSC IOVs OK"
else
    echo "L1-O2O-ERROR: TSC IOVs not OK!" >&2
    exit ${o2ocode}
fi

if [ ${nflag} -eq 0 ]
    then
    echo "`date` : writing RS payloads and setting RS IOVs"
    $CMSSW_BASE/src/CondTools/L1Trigger/scripts/runL1-O2O-rs.sh 10 ${tagbase}
    o2ocode=$?

    if [ ${o2ocode} -eq 0 ]
	then
	echo "L1-O2O-INFO: RS OK"
    else
	echo "L1-O2O-ERROR: RS not OK!" >&2
	exit ${o2ocode}
    fi
else
    echo "`date` : using default RS payloads"
fi

rm -f authentication.xml

exit ${o2ocode}
