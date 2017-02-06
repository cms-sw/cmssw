#!/bin/sh

xflag=0
CMS_OPTIONS=""
KEY_CONTENT=""

while getopts 'xfk:h' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      f) CMS_OPTIONS=$CMS_OPTIONS" forceUpdate=1"
          ;;
      k) KEY_CONTENT=$KEY_CONTENT" subsystemLabels=$OPTARG"
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
rskey=$3

echo "INFO: ADDITIONAL CMS OPTIONS:  " $CMS_OPTIONS

if [ ${xflag} -eq 0 ]
then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    INDB_OPTIONS="inputDBConnect=sqlite_file:l1config.db inputDBAuth=/data/O2O/L1T/" 
    OUTDB_OPTIONS="outputDBConnect=sqlite_file:l1config.db outputDBAuth=/data/O2O/L1T/" 
    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=sqlite_file:l1config.db"
#    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS copyDBAuth=/data/O2O/L1T/"
#    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS copyDBAuth=/data/O2O/L1T/"
else
#    echo "Writing to cms_orcoff_prep"
    echo "Writing to cms_orcon_prod"
#    INDB_OPTIONS="inputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS inputDBAuth=/data/O2O/L1T/"
#    OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS outputDBAuth=/data/O2O/L1T/"
    INDB_OPTIONS="inputDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS inputDBAuth=/data/O2O/L1T/"
    OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS outputDBAuth=/data/O2O/L1T/"
    #echo "Cowardly refusing to write to the online database"
    #exit
fi

#export UTM_XSD_DIR=/data/O2O/L1T/v9_20160823/CMSSW_8_0_18/utm/tmXsd

if cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyListExt=1 | grep "${tsckey}:${rskey}" ; then echo "TSC payloads present"
else
    echo "TSC payloads absent; writing $KEY_CONTENT now"
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWritePayloadOnlineExt_cfg.py tscKey=${tsckey} rsKey=${rskey} ${OUTDB_OPTIONS} ${COPY_OPTIONS} ${KEY_CONTENT} logTransactions=0 print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
    then
	echo "L1-O2O-ERROR: could not write TSC payloads"
	echo "L1-O2O-ERROR: could not write TSC payloads" 1>&2
	exit ${o2ocode}
    fi
fi

cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/L1ConfigWriteIOVOnlineExt_cfg.py ${CMS_OPTIONS} tscKey=${tsckey} rsKey=${rskey} runNumber=${runnum} ${OUTDB_OPTIONS} logTransactions=0 print | grep -Ev "CORAL.*Info|CORAL.*Debug"
o2ocode=${PIPESTATUS[0]}

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



