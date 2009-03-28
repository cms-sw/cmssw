#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
#export TNS_ADMIN=/nfshome0/xiezhen/conddb
export TNS_ADMIN=/nfshome0/l1emulator/o2o/conddb

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
    cmscond_bootstrap_detector -D L1T -f /nfshome0/l1emulator/o2o/conddb/dbconfigSqlite.xml -b $CMSSW_BASE
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/init_cfg.py tagBase=${tagbase} orconConnect=sqlite_file:l1config.db orconAuth=.
else
    echo "Setting up cms_orcoff_prep/CMS_COND_L1T account"
    cmscond_bootstrap_detector -D L1T -P /nfshome0/l1emulator/o2o/conddb -f /nfshome0/l1emulator/o2o/conddb/dbconfigInt2r.xml -b $CMSSW_BASE
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/init_cfg.py tagBase=${tagbase} orconConnect=oracle://cms_orcoff_prep/CMS_COND_L1T orconAuth=/nfshome0/l1emulator/o2o/conddb
#    echo "Setting up ORCON CMS_COND_21X_L1T account"
#    cmscond_bootstrap_detector -D L1T -f /nfshome0/o2o/conddb/dbconfigORCON.xml -b $CMSSW_BASE
#    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/init_cfg.py tagBase=${tagbase} orconConnect=oracle://cms_orcon_prod/CMS_COND_21X_L1T orconAuth=/nfshome0/xiezhen/conddb
fi

exit
