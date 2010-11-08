#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/popcondev/conddb

xflag=0
while getopts 'xh' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      h) echo "Usage: [-x] runnum tsckey tagbase"
          echo "  -x: write to ORCON instead of sqlite file"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
tsckey=$2
tagbase=$3

if [ ${xflag} -eq 0 ]
    then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py tagBase=${tagbase}_hlt inputDBConnect=sqlite_file:l1config.db inputDBAuth=. printL1TriggerKeyList=1 | grep ${tsckey} ; then echo "TSC payloads present"
    else
	echo "TSC payloads absent; writing now"
	cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=. logTransactions=0 print
	o2ocode=$?
	if [ ${o2ocode} -ne 0 ]
	    then
	    echo "L1-O2O-ERROR: could not write TSC payloads" >&2
	    exit ${o2ocode}
	fi
    fi

    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVOnline_cfg.py tscKey=${tsckey} runNumber=${runnum} tagBase=${tagbase}_hlt outputDBConnect=sqlite_file:l1config.db outputDBAuth=. logTransactions=0 print
    o2ocode=$?
    if [ ${o2ocode} -eq 0 ]
	then
	echo
	echo "`date` : checking O2O"
	if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py tagBase=${tagbase}_hlt inputDBConnect=sqlite_file:l1config.db inputDBAuth=. printL1TriggerKey=1 runNumber=${runnum} | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
	else
	    echo "L1-O2O-ERROR: IOV NOT OK" >&2
	    exit 199
	fi
    else
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." >&2
        else
            if [ ${o2ocode} -eq 65 ]
                then
                echo "L1-O2O-ERROR: problem writing object to ORCON." >&2
            fi
        fi
	exit ${o2ocode}
    fi
else
    if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py tagBase=${tagbase}_hlt inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb printL1TriggerKeyList=1 | grep ${tsckey} ; then echo "TSC payloads present"
    else
        echo "TSC payloads absent; writing now"
	cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb print
	o2ocode=$?
	if [ ${o2ocode} -ne 0 ]
	    then
	    echo "L1-O2O-ERROR: could not write TSC payloads" >&2
	    exit ${o2ocode}
	fi
    fi

    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVOnline_cfg.py tscKey=${tsckey} runNumber=${runnum} tagBase=${tagbase}_hlt outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb print
    o2ocode=$?
    if [ ${o2ocode} -eq 0 ]
	then
	echo
	echo "`date` : checking O2O"
	if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py tagBase=${tagbase}_hlt inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb printL1TriggerKey=1 runNumber=${runnum} | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
	else
	    echo "L1-O2O-ERROR: IOV NOT OK" >&2
	    exit 199
	fi
    else
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." >&2
        else
            if [ ${o2ocode} -eq 65 ]
                then
                echo "L1-O2O-ERROR: problem writing object to ORCON." >&2
            fi
        fi
	exit ${o2ocode}
    fi
fi
exit
