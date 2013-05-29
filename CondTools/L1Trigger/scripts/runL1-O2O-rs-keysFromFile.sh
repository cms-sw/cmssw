#!/bin/sh

xflag=0
oflag=0
pflag=0
while getopts 'xoph' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      o) oflag=1
          ;;
      p) pflag=1
	  ;;
      h) echo "Usage: [-x] tsckey runnum"
          echo "  -x: write to ORCON instead of sqlite file"
	  echo "  -o: overwrite keys"
          echo "  -p: centrally installed release, not on local machine"
	  exit
          ;;
  esac
done
shift $(($OPTIND - 1))

if [ ${pflag} -eq 0 ]
    then
    export SCRAM_ARCH=""
    export VO_CMS_SW_DIR=""
    source /opt/cmssw/cmsset_default.sh
else
    source /nfshome0/cmssw2/scripts/setup.sh
fi
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/popcondev/conddb

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
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSPayloadOnline_cfg.py outputDBConnect=sqlite_file:l1config.db outputDBAuth=. keysFromDB=0 ${line} ${overwrite} logTransactions=0 print
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSIOVOnline_cfg.py outputDBConnect=sqlite_file:l1config.db outputDBAuth=. keysFromDB=0 ${line} logTransactions=0 print
  else
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSPayloadOnline_cfg.py outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb keysFromDB=0 ${line} ${overwrite} print
      cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSIOVOnline_cfg.py outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb keysFromDB=0 ${line} print
  fi
done

exit
