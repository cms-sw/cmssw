#!/bin/sh

if [ $# -ne 2 ] ; then echo 'Please, provide the 2 arguments: tscKey and rsKey'; exit 2; fi

# setup the environment at p5
source /opt/offline/cmsset_default.sh
cd /opt/offline/slc6_amd64_gcc493/cms/cmssw/CMSSW_8_0_25
cmsenv
cd -
export TNS_ADMIN=/opt/offline/slc6_amd64_gcc493/cms/oracle-env/29/etc/
# path to the .cms_cond DB access token
DBAuth=/data/O2O/L1T/
# speed-up by using pre-initialized sqlite file (it'll be created locally if not present)
blankDB=/data/O2O/L1T/l1configBlank.db

## or setup the environment on lxplus:
#cd /cvmfs/cms.cern.ch/slc6_amd64_gcc530/cms/cmssw/CMSSW_9_0_2
#eval `scram runtime -sh`
#cd -
#export TNS_ADMIN=/cvmfs/cms.cern.ch/slc6_amd64_gcc493/cms/oracle-env/29/etc/
## path to the .cms_cond DB access token
#DBAuth=/afs/cern.ch/user/k/kkotov/
#blankDB=/afs/cern.ch/user/k/kkotov/public/l1configBlank.db

# Initialize the sqlite file and save there constant payloads
rm -f l1config.db
if [ -e $blankDB ] ; then
    echo "Using pre-initialized $blankDB sqlite file";
    cp $blankDB l1config.db
else
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/init_cfg.py useO2OTags=1 outputDBConnect=sqlite:l1configBlank.db outputDBAuth=${DBAuth}
    initcode=$?
    if [ $initcode -ne 0 ] ; then echo "Failed to initialize sqlite file"; exit 1 ; fi

    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWriteSinglePayloadExt_cfg.py objectKey="OMTF_ALGO_EMPTY" objectType=L1TMuonOverlapParams recordName=L1TMuonOverlapParamsO2ORcd useO2OTags=1 outputDBConnect=sqlite:l1configBlank.db outputDBAuth=${DBAuth}
    initcode=$?
    if [ $initcode -ne 0 ] ; then echo "Failed to write OMTF_ALGO_EMPTY constant payload in sqlite file" ; exit 1 ; fi
    cmsRun ${CMSSW_BASE}/src/CondTools/L1TriggerExt/test/L1ConfigWriteSinglePayloadExt_cfg.py objectKey="1541" objectType=L1TMuonEndCapForest recordName=L1TMuonEndcapForestO2ORcd useO2OTags=1 outputDBConnect=sqlite:l1configBlank.db
    initcode=$?
    if [ $initcode -ne 0 ] ; then echo "Failed to write EMTF pT LUT #1541 in sqlite file" ; exit 1 ; fi

    cp l1configBlank.db l1config.db
fi

# get all subsystem keys from the top-level TSC and RS keys
keys=$(cmsRun ${CMSSW_BASE}/src/L1TriggerConfig/Utilities/test/viewTKEonline.py tscKey=$1 rsKey=$2 DBAuth=${DBAuth} 2>/dev/null | grep ' key: ')
keyscode=$?
if [ $keyscode -ne 0 ] ; then echo "Failed to get the list of trigger keys for L1T subsystems. Did you provide the 2 arguments: tscKey and rsKey ? " ; exit 2 ; fi

# split the keys above
uGT_key=$(echo $keys | sed -n -e's|.*uGT *key: \([^ ]*\).*|\1|gp')
uGMT_key=$(echo $keys | sed -n -e's|.*uGMT *key: \([^ ]*\).*|\1|gp')
CALO_key=$(echo $keys | sed -n -e's|.*CALO *key: \([^ ]*\).*|\1|gp')
BMTF_key=$(echo $keys | sed -n -e's|.*BMTF *key: \([^ ]*\).*|\1|gp')
OMTF_key=$(echo $keys | sed -n -e's|.*OMTF *key: \([^ ]*\).*|\1|gp')
echo "uGT_key=$uGT_key uGMT_key=$uGMT_key CALO_key=$CALO_key BMTF_key=$BMTF_key OMTF_key=$OMTF_key"

# go one-by-one over the systems
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

# check if any of the processes above failed
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
