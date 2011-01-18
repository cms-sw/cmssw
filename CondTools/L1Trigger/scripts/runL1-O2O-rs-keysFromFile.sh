#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/popcondev/conddb

xflag=0
oflag=0
while getopts 'xoh' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      o) oflag=1
          ;;
      h) echo "Usage: [-x] tsckey runnum tagbase"
          echo "  -x: write to ORCON instead of sqlite file"
	  echo "  -o: overwrite keys"
	  exit
          ;;
  esac
done
shift $(($OPTIND - 1))

tagbase=$1

if [ ${oflag} -eq 1 ]
    then
    overwrite="overwriteKeys=1"
fi

exec<rskeys.txt

while read line
do

  if [ ${xflag} -eq 0 ]
      then
      echo "Writing to sqlite_file:l1config.db instead of ORCON."
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSPayloadOnline_cfg.py tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=. keysFromDB=0 ${line} ${overwrite} logTransactions=0 print
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSIOVOnline_cfg.py tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=. keysFromDB=0 ${line} logTransactions=0 print
  else
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSPayloadOnline_cfg.py tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb keysFromDB=0 ${line} ${overwrite} print
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSIOVOnline_cfg.py tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb keysFromDB=0 ${line} print
  fi
done

exit
