#!/bin/sh

xflag=0
CMS_OPTIONS=""

while getopts 'xfh' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      f) CMS_OPTIONS=$CMS_OPTIONS" forceUpdate=1"
          ;;
      h) echo "Usage: [-xf] runnum tsckey"
          echo "  -x: write to ORCON instead of sqlite file"
          echo "  -f: force IOV update"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
tsckey=$2

echo "INFO: ADDITIONAL CMS OPTIONS:  " $CMS_OPTIONS

if [ ${xflag} -eq 0 ]
then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    INDB_OPTIONS="inputDBConnect=sqlite_file:l1config.db inputDBAuth=." 
    OUTDB_OPTIONS="outputDBConnect=sqlite_file:l1config.db outputDBAuth=." 
    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=sqlite_file:l1config.db"
#    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS copyDBAuth=/data/O2O/L1T/pro/o2o/"
#    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS copyDBAuth=/data/O2O/L1T/pro/o2o/"
else
    echo "Writing to cms_orcoff_prep"
#    INDB_OPTIONS="inputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS inputDBAuth=/data/O2O/L1T/pro/o2o/"
#    OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS outputDBAuth=/data/O2O/L1T/pro/o2o/"
    INDB_OPTIONS="inputDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS inputDBAuth=/data/O2O/L1T/pro/o2o/"
    OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS outputDBAuth=/data/O2O/L1T/pro/o2o/"
    #echo "Cowardly refusing to write to the online database"
    #exit
fi


if cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyListExt=1 | grep ${tsckey} ; then echo "TSC payloads present"
else
    echo "TSC payloads absent; writing now"
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWritePayloadOnlineExt_cfg.py tscKey=${tsckey} ${OUTDB_OPTIONS} ${COPY_OPTIONS} logTransactions=0 print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
    then
	echo "L1-O2O-ERROR: could not write TSC payloads"
	echo "L1-O2O-ERROR: could not write TSC payloads" 1>&2
	exit ${o2ocode}
    fi
fi

cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/L1ConfigWriteIOVOnlineExt_cfg.py ${CMS_OPTIONS} tscKey=${tsckey} runNumber=${runnum} ${OUTDB_OPTIONS} logTransactions=0 print
o2ocode=$?

if [ ${o2ocode} -eq 0 ]
then
    echo
    echo "`date` : checking O2O"
    if cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyExt=1 runNumber=${runnum} | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
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



