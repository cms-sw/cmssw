#!/bin/sh

xflag=0
cflag=0
CMS_OPTIONS=""

while getopts 'xoch' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      o) CMS_OPTIONS=$CMS_OPTIONS" overwriteKeys=1"
          ;;
      c) cflag=1
          ;;
      h) echo "Usage: [-x] [-o] tsckey"
	  echo "  -x: write to ORCON instead of sqlite file"
	  echo "  -o: overwrite keys"
	  echo "  -c: copy non-O2O payloads from ORCON"
	  exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

tsckey=$1
shift 1

COPY_OPTIONS=""
if [ ${cflag} -eq 1 ]
then
    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS copyDBAuth=/nfshome0/l1emulator/run2/o2o/prod"
#    COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS copyDBAuth=/nfshome0/l1emulator/run2/o2o/v1"
fi

if [ ${xflag} -eq 0 ]
then
    echo "Writing to sqlite_file:l1config.db instead of ORCON."
    INDB_OPTIONS="inputDBConnect=sqlite_file:l1config.db inputDBAuth=."
    OUTDB_OPTIONS="outputDBConnect=sqlite_file:l1config.db outputDBAuth=." 
else
    echo "Writing to cms_orcoff_prep"
#    INDB_OPTIONS="inputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS inputDBAuth=/nfshome0/l1emulator/run2/o2o/v1"
#    OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS outputDBAuth=/nfshome0/l1emulator/run2/o2o/v1"
    #echo "Cowardly refusing to write to the online database"
    INDB_OPTIONS="inputDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS inputDBAuth=/nfshome0/l1emulator/run2/o2o/prod"
    OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS outputDBAuth=/nfshome0/l1emulator/run2/o2o/prod"  
    #exit
fi

cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/L1ConfigWritePayloadOnlineExt_cfg.py tscKey=${tsckey} ${CMS_OPTIONS} ${OUTDB_OPTIONS} ${COPY_OPTIONS} logTransactions=0 print
o2ocode=$?
if [ ${o2ocode} -eq 0 ]
then
    echo
    echo "`date` : checking O2O"
    if cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyListExt=1 | grep ${tsckey} ; then echo "L1TRIGGERKEY WRITTEN SUCCESSFULLY"
    else
	echo "L1-O2O-ERROR: L1TRIGGERKEY WRITING FAILED"
	echo "L1-O2O-ERROR: L1TRIGGERKEY WRITING FAILED" 1>&2
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
