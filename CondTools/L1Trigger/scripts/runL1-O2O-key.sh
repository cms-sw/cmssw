#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/popcondev/conddb_cmsusr

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
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=. print
    echo
    echo "`date` : checking O2O"
    if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py tagBase=${tagbase}_hlt inputDBConnect=sqlite_file:l1config.db inputDBAuth=. printL1TriggerKeyList=1 | grep ${tsckey} ; then echo "L1TRIGGERKEY WRITTEN SUCCESSFULLY"
    else
        echo "L1TRIGGERKEY WRITING FAILED"
        exit 199
    fi
else
    echo "Writing to cms_orcon_prod."
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb print
    echo
    echo "`date` : checking O2O"
    if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py tagBase=${tagbase}_hlt inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb printL1TriggerKeyList=1 | grep ${tsckey} ; then echo "L1TRIGGERKEY WRITTEN SUCCESSFULLY"
    else
	echo "L1TRIGGERKEY WRITING FAILED"
	exit 199
    fi
fi

exit
