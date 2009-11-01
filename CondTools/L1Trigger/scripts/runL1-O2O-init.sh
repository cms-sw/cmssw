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
      h) echo "Usage: [-x] tagbase"
          echo "  -x: write to ORCON instead of sqlite file"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

tagbase=$1

if [ ${xflag} -eq 0 ]
    then
    echo "Setting up sqlite_file:l1config.db"
    cmscond_bootstrap_detector -D L1T -f $CMSSW_BASE/src/CondTools/L1Trigger/test/dbconfiguration.xml -b $CMSSW_BASE
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/init_cfg.py tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=.
else
    echo "Setting up cms_orcon_prod/CMS_COND_L1T account"
    cmscond_bootstrap_detector -D L1T -P /nfshome0/popcondev/conddb -f /nfshome0/popcondev/L1Job/conddb/dbconfigORCON.xml -b $CMSSW_BASE
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/init_cfg.py tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb
fi

exit
