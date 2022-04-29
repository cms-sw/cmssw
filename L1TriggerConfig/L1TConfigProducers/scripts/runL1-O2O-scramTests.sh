#!/bin/sh
#set -x

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
          echo "  -d: dron these systems form the job: Dont create WriterProxyT for these systems in the PayloadWriter Constructor (e.g. -d EMTF,OMTF,CALO)"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

runnum=$1
tsckey=$2
rskey=$3


echo CMSSW_BASE = $CMSSW_BASE
echo PWD = $PWD
ls -a

export TNS_ADMIN=/etc
echo "INFO: ADDITIONAL CMS OPTIONS:  " $CMS_OPTIONS $KEY_CONTENT $TAG_UPDATE

ONLINEDB_OPTIONS="onlineDBConnect=oracle://cms_omds_adg/CMS_TRG_R onlineDBAuth=$HOME/"
PROTODB_OPTIONS="protoDBConnect=oracle://cms_orcon_adg/CMS_CONDITIONS protoDBAuth=$HOME/"

## test dir
DATA_DIR=$CMSSW_BASE/$SCRAM_TEST_NAME
mkdir $DATA_DIR
## get sqlite from the write-only CMSSW_SEARCH_PATH
for dir in $(echo $CMSSW_SEARCH_PATH | tr ':' '\n') ; do
  [ -e $dir/L1TriggerConfig/L1TConfigProducers/data/l1config.db ] || continue
  cp $dir/L1TriggerConfig/L1TConfigProducers/data/l1config.db $DATA_DIR/
  break
done
local_db=$DATA_DIR/l1config.db
sqlite3 $local_db -cmd "SELECT * from IOV;" ".q" > $DATA_DIR/iov_before

echo "Writing to sqlite_file:$local_db instead of ORCON."
INDB_OPTIONS="inputDBConnect=sqlite_file:$local_db inputDBAuth=$HOME/"
OUTDB_OPTIONS="outputDBConnect=sqlite_file:$local_db outputDBAuth=$HOME/"
COPY_OPTIONS="copyNonO2OPayloads=1 copyDBConnect=sqlite_file:$local_db"

##
## =============== O2O Job ===============
##

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
    if cmsRun $CMSSW_BASE/src/CondTools/L1TriggerExt/test/l1o2otestanalyzer_cfg.py ${INDB_OPTIONS} printL1TriggerKeyExt=1 runNumber=${runnum} ${TAG_UPDATE} | c++filt --types | grep ${tsckey} ; then echo "L1-O2O-INFO: IOV OK"
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

echo ">>>>>>>"
echo ">>>>>>> JOB REPORT"
echo "============================== BEFORE O2O =============================="
cat $DATA_DIR/iov_before
echo "============================== AFTER O2O =============================="
sqlite3 $local_db -cmd "SELECT * from IOV;" ".q"

rm -r $DATA_DIR
exit $o2ocode

