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

runnum=$1
tagbase=$2
l1Key=$3

if [ ${oflag} -eq 1 ]
    then
    overwrite="overwriteKeys=1"
fi

if [ ${xflag} -eq 0 ]
    then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=. ${overwrite} logTransactions=0 `$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getKeys.sh -r ${l1Key}` print
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
else
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteRSOnline_cfg.py runNumber=${runnum} tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb ${overwrite} `$CMSSW_BASE/src/CondTools/L1Trigger/scripts/getKeys.sh -r ${l1Key}` print
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
fi

exit ${o2ocode}
