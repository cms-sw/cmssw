#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/xiezhen/conddb

xflag=0
while getopts 'xh' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      h) echo "Usage: [-x] tsckey tagbase"
	  echo "  -x: write to ORCON instead of sqlite file"
	  exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

tsckey=$1
tagbase=$2
shift 2

if [ ${xflag} -eq 0 ]
    then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} tagBase=${tagbase} orconConnect=sqlite_file:l1config.db orconAuth=. print
else
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} tagBase=${tagbase} orconConnect=oracle://cms_orcon_prod/CMS_COND_21X_L1T orconAuth=/nfshome0/xiezhen/conddb print
fi

exit
