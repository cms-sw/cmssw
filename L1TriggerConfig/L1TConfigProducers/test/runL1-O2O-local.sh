#!/bin/sh
set -x

xflag=0
CMS_OPTIONS=""
KEY_CONTENT=""
TAG_UPDATE=""
UNSAFE=""

while getopts 'xfk:t:u:d:h' OPTION
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
      u) if [ -z $UNSAFE ] ; then UNSAFE="unsafe="; else UNSAFE=$UNSAFE","; fi
         UNSAFE=$UNSAFE$OPTARG
          ;;
      d) if [ -z $DROPSYSTEMS ] ; then DROPSYSTEMS="dropFromJob="; else DROPSYSTEMS=$DROPSYSTEMS","; fi
         DROPSYSTEMS=$DROPSYSTEMS$OPTARG
         ;; 
      h) echo "Usage: [-xf] runnum tsckey"
          echo "  -x: write to ORCON instead of sqlite file"
          echo "  -f: force IOV update"
          echo "  -k: limit update to the specific systems (default are all, which is equivalent to -k uGT,uGTrs,GMT,EMTF,OMTF,BMTF,CALO)"
          echo "  -t: override tag name as TYPE:NEW_TAG_BASE (e.g. -t L1TCaloParams:Stage2v1)"
          echo "  -u: lift transaction safety: carry on even problems are encountered (e.g. -u EMTF,OMTF,CALO)"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
tsckey=$2
rskey=$3

#CMSSW_BASE=${RELEASEDIR}
#CMSSW_BASE=${JOBDIR}/${RELEASE} #for local CMSSW checkout
echo CMSSW_BASE = $CMSSW_BASE

export TNS_ADMIN=/etc

echo "INFO: ADDITIONAL CMS OPTIONS:  " $CMS_OPTIONS $KEY_CONTENT $TAG_UPDATE

ONLINEDB_OPTIONS="onlineDBConnect=oracle://cms_omds_adg/CMS_TRG_R onlineDBAuth=./"
#ONLINEDB_OPTIONS="onlineDBAuth=./"
PROTODB_OPTIONS="protoDBConnect=oracle://cms_orcon_adg/CMS_CONDITIONS protoDBAuth=./"

local_db=L1TriggerConfig/L1TConfigProducers/data/l1config.db
echo "Writing to sqlite_file:$local_db instead of ORCON."
INDB_OPTIONS="inputDBConnect=sqlite_file:$local_db inputDBAuth=./" 
OUTDB_OPTIONS="outputDBConnect=sqlite_file:$local_db outputDBAuth=./"
COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=sqlite_file:$local_db"

# echo "Writing to cms_orcoff_prep instead of ORCON"
# INDB_OPTIONS="inputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS inputDBAuth=/data/O2O/L1T/"
# OUTDB_OPTIONS="outputDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS outputDBAuth=/data/O2O/L1T/"
# COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=oracle://cms_orcoff_prep/CMS_CONDITIONS copyDBAuth=/data/O2O/L1T/"



if cmsRun -e ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyListExt=1 ${TAG_UPDATE} | c++filt --types | grep "${tsckey}:${rskey}" ; then echo "TSC payloads present"
else
    echo "TSC payloads absent; writing $KEY_CONTENT now"
    cmsRun -e ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWritePayloadOnlineExt_cfg.py tscKey=${tsckey} rsKey=${rskey} ${ONLINEDB_OPTIONS} ${PROTODB_OPTIONS} ${OUTDB_OPTIONS} ${COPY_OPTIONS} ${KEY_CONTENT} ${TAG_UPDATE} ${UNSAFE} ${DROPSYSTEMS} logTransactions=0 print | c++filt --types | tee -a lastLogForFM.txt
    #cmsRun ./L1ConfigWritePayloadOnlineExt_cfg.py tscKey=${tsckey} rsKey=${rskey} ${OUTDB_OPTIONS1} ${COPY_OPTIONS} ${KEY_CONTENT} ${TAG_UPDATE} ${UNSAFE} logTransactions=0 print | tee -a lastLogForFM.txt
    o2ocode=${PIPESTATUS[0]}
#    o2ocode=$?
    if [ ${o2ocode} -ne 0 ]
    then
	echo "L1-O2O-ERROR: could not write TSC payloads"
	echo "L1-O2O-ERROR: could not write TSC payloads" 1>&2
	exit ${o2ocode}
    fi
fi

cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/L1ConfigWriteIOVOnlineExt_cfg.py ${CMS_OPTIONS} tscKey=${tsckey} rsKey=${rskey} runNumber=${runnum} ${OUTDB_OPTIONS} ${TAG_UPDATE} logTransactions=0 print | grep -Ev "CORAL.*Info|CORAL.*Debug" | c++filt --types | tee -a lastLogForFM.txt
o2ocode=${PIPESTATUS[0]}

if [ ${o2ocode} -eq 0 ]
then
    echo
    echo "`date` : checking O2O"
    if cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyExt=1 runNumber=${runnum} ${TAG_UPDATE} | grep ${tsckey} | c++filt --types ; then echo "L1-O2O-INFO: IOV OK"
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



