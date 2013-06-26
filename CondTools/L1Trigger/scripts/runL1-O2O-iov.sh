#!/bin/sh

xflag=0
pflag=0
while getopts 'xph' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      p) pflag=1
	  ;;
      h) echo "Usage: [-x] runnum tsckey"
          echo "  -x: write to ORCON instead of sqlite file"
          echo "  -p: centrally installed release, not on local machine"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
tsckey=$2

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

if [ ${xflag} -eq 0 ]
    then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py inputDBConnect=sqlite_file:l1config.db inputDBAuth=. printL1TriggerKeyList=1 | grep ${tsckey} ; then echo "TSC payloads present"
    else
	echo "TSC payloads absent; writing now"
	cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} outputDBConnect=sqlite_file:l1config.db outputDBAuth=. logTransactions=0 print
	o2ocode=$?
	if [ ${o2ocode} -ne 0 ]
	    then
	    echo "L1-O2O-ERROR: could not write TSC payloads"
	    echo "L1-O2O-ERROR: could not write TSC payloads" 1>&2
	    exit ${o2ocode}
	fi
    fi

    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVOnline_cfg.py tscKey=${tsckey} runNumber=${runnum} outputDBConnect=sqlite_file:l1config.db outputDBAuth=. logTransactions=0 print
    o2ocode=$?
    if [ ${o2ocode} -eq 0 ]
	then
	echo
	echo "`date` : checking O2O"
	if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py inputDBConnect=sqlite_file:l1config.db inputDBAuth=. printL1TriggerKey=1 runNumber=${runnum} | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
	else
	    echo "L1-O2O-ERROR: IOV NOT OK"
	    echo "L1-O2O-ERROR: IOV NOT OK" 1>&2
	    exit 199
	fi
    else
	if [ ${o2ocode} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)."
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." 1>&2
        else
            if [ ${o2ocode} -eq 65 ]
                then
                echo "L1-O2O-ERROR: problem writing object to ORCON."
                echo "L1-O2O-ERROR: problem writing object to ORCON." 1>&2
            fi
        fi
	exit ${o2ocode}
    fi
else
    echo "`date` : checking for TSC payloads"
#    if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb_taskWriters/L1T printL1TriggerKeyList=1 | grep ${tsckey} ; then echo "`date` : TSC payloads present"
#    else
        echo "`date` : TSC payloads absent; writing now"
	cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWritePayloadOnline_cfg.py tscKey=${tsckey} outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb_taskWriters/L1T print
	o2ocode1=$?
	if [ ${o2ocode1} -ne 0 ]
	    then
	    echo "L1-O2O-ERROR: could not write TSC payloads"
	    echo "L1-O2O-ERROR: could not write TSC payloads" 1>&2
#	    exit ${o2ocode1}
	fi
#    fi

    echo "`date` : setting TSC IOVs"
    cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/L1ConfigWriteIOVOnline_cfg.py tscKey=${tsckey} runNumber=${runnum} outputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T outputDBAuth=/nfshome0/popcondev/conddb_taskWriters/L1T print
    o2ocode2=$?
    if [ ${o2ocode2} -eq 0 ]
	then
	echo
	echo "`date` : checking O2O"
	if cmsRun $CMSSW_BASE/src/CondTools/L1Trigger/test/l1o2otestanalyzer_cfg.py inputDBConnect=oracle://cms_orcon_prod/CMS_COND_31X_L1T inputDBAuth=/nfshome0/popcondev/conddb_taskWriters/L1T printL1TriggerKey=1 runNumber=${runnum} | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
	else
	    echo "L1-O2O-ERROR: IOV NOT OK"
	    echo "L1-O2O-ERROR: IOV NOT OK" 1>&2
	    exit 199
	fi
    else
	if [ ${o2ocode2} -eq 66 ]
	    then
	    echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check that /nfshome0/centraltspro/secure/authentication.xml is up to date (OMDS)." 1>&2
        else
            if [ ${o2ocode2} -eq 65 ]
                then
                echo "L1-O2O-ERROR: problem writing object to ORCON."
                echo "L1-O2O-ERROR: problem writing object to ORCON." 1>&2
            fi
        fi
#	exit ${o2ocode}
    fi

    o2ocode=`echo ${o2ocode1} + ${o2ocode2} | bc`
    exit ${o2ocode}
fi
exit
