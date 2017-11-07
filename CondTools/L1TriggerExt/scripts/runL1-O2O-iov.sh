#!/bin/sh

xflag=0
CMS_OPTIONS=""
KEY_CONTENT=""
TAG_UPDATE=""
UNSAFE=""

while getopts 'xfk:t:s:h' OPTION
  do
  case $OPTION in
      x) xflag=1
          ;;
      f) CMS_OPTIONS=$CMS_OPTIONS" forceUpdate=1"
          ;;
      k) KEY_CONTENT=$KEY_CONTENT" subsystemLabels=$OPTARG"
          ;;
      t) if [ -z $TAG_UPDATE ] ; then TAG_UPDATE="tagUpdate="; else TAG_UPDATE=$TAG_UPDATE","; fi
         TAG_UPDATE=$TAG_UPDATE$OPTARG
          ;;
      s) if [ -z $UNSAFE ] ; then UNSAFE="unsafe="; else UNSAFE=$UNSAFE","; fi
         UNSAFE=$UNSAFE$OPTARG
          ;;
      h) echo "Usage: [-xf] runnum tsckey"
          echo "  -x: write to ORCON instead of sqlite file"
          echo "  -f: force IOV update"
          echo "  -k: limit update to the specific systems (default are all, which is equivalent to -k uGT,uGTrs,GMT,EMTF,OMTF,BMTF,CALO)"
          echo "  -t: override tag name as TYPE:NEW_TAG_BASE (e.g. -t L1TCaloParams:Stage2v1)"
          echo "  -s: lift transaction safety: carry on even problems are encountered (e.g. -s EMTF,OMTF,CALO)"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
tsckey=$2
rskey=$3

export TNS_ADMIN=/opt/offline/slc6_amd64_gcc493/cms/oracle-env/29/etc/

echo "INFO: ADDITIONAL CMS OPTIONS:  " $CMS_OPTIONS $KEY_CONTENT $TAG_UPDATE

ONLINEDB_OPTIONS="onlineDBConnect=oracle://cms_omds_adg/CMS_TRG_R onlineDBAuth=./"
PROTODB_OPTIONS="protoDBConnect=oracle://cms_orcon_adg/CMS_CONDITIONS protoDBAuth=./"
#ONLINEDB_OPTIONS="onlineDBConnect=oracle://cms_omds_lb/CMS_TRG_R onlineDBAuth=/data/O2O/L1T/"
#PROTODB_OPTIONS="protoDBConnect=oracle://cms_orcon_prod/CMS_CONDITIONS protoDBAuth=/data/O2O/L1T/"

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

if cmsRun ${CMSSW_RELEASE_BASE}/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyListExt=1 | grep "${tsckey}:${rskey}" ; then echo "TSC payloads present"
else
    echo "TSC payloads absent; writing $KEY_CONTENT now"
    cmsRun ${CMSSW_RELEASE_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWritePayloadOnlineExt_cfg.py tscKey=${tsckey} rsKey=${rskey} ${ONLINEDB_OPTIONS} ${PROTODB_OPTIONS} ${OUTDB_OPTIONS} ${COPY_OPTIONS} ${KEY_CONTENT} ${TAG_UPDATE} ${UNSAFE} logTransactions=0 print
    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
    then
	echo "L1-O2O-ERROR: could not write TSC payloads"
	echo "L1-O2O-ERROR: could not write TSC payloads" 1>&2
	exit ${o2ocode}
    fi
fi

cmsRun $CMSSW_RELEASE_BASE/src/CondTools/L1TriggerExt/test/L1ConfigWriteIOVOnlineExt_cfg.py ${CMS_OPTIONS} tscKey=${tsckey} rsKey=${rskey} runNumber=${runnum} ${OUTDB_OPTIONS} ${TAG_UPDATE} logTransactions=0 print | grep -Ev "CORAL.*Info|CORAL.*Debug"
o2ocode=${PIPESTATUS[0]}

if [ ${o2ocode} -eq 0 ]
then
    echo
    echo "`date` : checking O2O"
    if cmsRun $CMSSW_RELEASE_BASE/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyExt=1 runNumber=${runnum} ${TAG_UPDATE} | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
    else
	echo "L1-O2O-ERROR: IOV NOT OK"
	echo "L1-O2O-ERROR: IOV NOT OK" 1>&2
	exit 199
    fi
else
    if [ ${o2ocode} -eq 66 ]
    then
	echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check authentication token .cms_cond/db.key"
	echo "L1-O2O-ERROR: unable to connect to OMDS or ORCON.  Check authentication token .cms_cond/db.key" 1>&2
    else
        if [ ${o2ocode} -eq 65 ]
        then
            echo "L1-O2O-ERROR: problem writing object to ORCON."
            echo "L1-O2O-ERROR: problem writing object to ORCON." 1>&2
        fi
    fi
    exit ${o2ocode}
fi



