#!/bin/sh

if [ $# -ne 2 ] ; then echo 'Please, provide the 2 arguments: tscKey and rsKey'; exit 2; fi

#source /opt/offline/cmsset_default.sh
#cd /data/O2O/L1T/v9.0/CMSSW_8_0_18
#cmsenv
#cd -

DBAuth=/data/O2O/L1T/

rm -f l1config.db

if [ -e l1configBlank.db ] ; then
    echo "Using pre-initialized l1configBlank.db sqlite file";
else
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/init_cfg.py useO2OTags=1 outputDBConnect=sqlite:./l1configBlank.db outputDBAuth=${DBAuth}
    initcode=$?
    if [ $initcode -ne 0 ] ; then echo "Failed to initialize sqlite file"; exit 1 ; fi
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWriteSinglePayloadExt_cfg.py objectKey="OMTF_ALGO_EMPTY" objectType=L1TMuonOverlapParams recordName=L1TMuonOverlapParamsO2ORcd useO2OTags=1 outputDBConnect=sqlite:l1configBlank.db outputDBAuth=${DBAuth}
    initcode=$?
    if [ $initcode -ne 0 ] ; then echo "Failed to write OMTF_ALGO_EMPTY in sqlite file" ; exit 1 ; fi
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWriteSinglePayloadExt_cfg.py objectKey="EMTF_ALGO_EMPTY" objectType=L1TMuonEndCapParams recordName=L1TMuonEndcapParamsO2ORcd useO2OTags=1 outputDBConnect=sqlite:l1configBlank.db outputDBAuth=${DBAuth}
    initcode=$?
    if [ $initcode -ne 0 ] ; then echo "Failed to write EMTF_ALGO_EMPTY in sqlite files" ; exit 1 ; fi
fi

# save some time by restoring an always ready template:
cp l1configBlank.db l1config.db

keys=$(cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/viewTKEonline.py tscKey=$1 rsKey=$2 DBAuth=${DBAuth} 2>/dev/null | grep ' key: ')

keyscode=$?
if [ $keyscode -ne 0 ] ; then echo "Failed to get the list of trigger keys for L1T subsystems. Did you provide the 2 arguments: tscKey and rsKey ? " ; exit 2 ; fi

uGT_key=$(echo $keys | sed -n -e's|.*uGT *key: \([^ ]*\).*|\1|gp')
uGMT_key=$(echo $keys | sed -n -e's|.*uGMT *key: \([^ ]*\).*|\1|gp')
CALO_key=$(echo $keys | sed -n -e's|.*CALO *key: \([^ ]*\).*|\1|gp')
BMTF_key=$(echo $keys | sed -n -e's|.*BMTF *key: \([^ ]*\).*|\1|gp')
OMTF_key=$(echo $keys | sed -n -e's|.*OMTF *key: \([^ ]*\).*|\1|gp')

echo "uGT_key=$uGT_key uGMT_key=$uGMT_key CALO_key=$CALO_key BMTF_key=$BMTF_key OMTF_key=$OMTF_key"

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TUtmTriggerMenu.py systemKey=$uGT_key DBAuth=${DBAuth}
ugtcode=$?

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TCaloParams.py systemKey=$CALO_key DBAuth=${DBAuth}
caloparcode=$?

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TMuonBarrelParams.py systemKey=$BMTF_key DBAuth=${DBAuth}
bmtfcode=$?

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TMuonGlobalParams.py systemKey=$uGMT_key DBAuth=${DBAuth}
ugmtcode=$?

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TGlobalPrescalesVetos.py systemKey=$uGT_key DBAuth=${DBAuth}
ugtrscode=$?

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TMuonOverlapParams.py topKey="$1:$2" DBAuth=${DBAuth}
omtfcode=$?

cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/dumpL1TMuonEndcapParams.py topKey="$1:$2" DBAuth=${DBAuth}
emtfcode=$?

exitcode=0

if [ $ugtcode     -ne 0 ] ; then exitcode=`expr $exitcode + 10`; fi
if [ $caloparcode -ne 0 ] ; then exitcode=`expr $exitcode + 100`; fi
if [ $bmtfcode    -ne 0 ] ; then exitcode=`expr $exitcode + 1000`; fi
if [ $ugmtcode    -ne 0 ] ; then exitcode=`expr $exitcode + 10000`; fi
if [ $ugtrscode   -ne 0 ] ; then exitcode=`expr $exitcode + 100000`; fi
if [ $omtfcode    -ne 0 ] ; then exitcode=`expr $exitcode + 1000000`; fi
if [ $emtfcode    -ne 0 ] ; then exitcode=`expr $exitcode + 10000000`; fi

echo "Status codes: uGT: $ugtcode,  CALO: $caloparcode,  BMTF: $bmtfcode,  uGMT: $ugmtcode,  uGTrs: $ugtrscode,  OMTF: $omtfcode,  EMTF:$emtfcode,  Total: $exitcode"

if [ $exitcode -eq 0 ] ; then
    echo "Everything looks good"
else
    echo "Problems encountered, NOT ok"
fi

exit $exitcode
