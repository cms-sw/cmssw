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
      h) echo "Usage: [-x] runnum"
          echo "  -x: write to ORCON instead of sqlite file"
	  echo "  -o: overwrite keys"
          echo "  -p: centrally installed release, not on local machine"
          exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1

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

if [ ${xflag} -eq 0 ]
    then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSPayloadOnline_cfg.py outputDBConnect=sqlite_file:l1config.db outputDBAuth=. ${overwrite} logTransactions=0 print
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSIOVOnline_cfg.py runNumber=${runnum} outputDBConnect=sqlite_file:l1config.db outputDBAuth=. logTransactions=0 print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
	then
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." >&2
	else
	    if [ ${o2ocode} -eq 65 ]
		then
		echo "L1-O2O-ERROR: problem writing object to ORCON." >&2
	    fi
	fi
    fi

#    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} outputDBConnect=sqlite_file:l1config.db outputDBAuth=. print
#    echo "`date` : checking O2O"
#    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py inputDBConnect=sqlite_file:l1config.db inputDBAuth=. printRSKeys=1 runNumber=${runnum}

else
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSPayloadOnline_cfg.py outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb ${overwrite} print
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSIOVOnline_cfg.py runNumber=${runnum} outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
	then
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." >&2
	else
	    if [ ${o2ocode} -eq 65 ]
		then
		echo "L1-O2O-ERROR: problem writing object to ORCON." >&2
	    fi
	fi
    fi

#    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb print
#    echo "`date` : checking O2O"
#    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb printRSKeys=1 runNumber=${runnum}

fi

exit ${o2ocode}
