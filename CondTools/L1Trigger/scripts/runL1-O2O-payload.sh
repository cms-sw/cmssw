#!/bin/sh

source /nfshome0/cmssw2/scripts/setup.sh
eval `scramv1 run -sh`
export TNS_ADMIN=/nfshome0/xiezhen/conddb

sflag=0
nflag=0
while getopts 'snh' OPTION
  do
  case $OPTION in
      s) sflag=1
          ;;
      n) nflag=1
          ;;
      h) echo "Usage: [-s] [-n] record type key"
	  echo "  -s: write to sqlite file instead of ORCON"
	  echo "  -n: generate .py file but do not run it"
	  exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

record=$1
type=$2
key=$3
tagbase=$4

if [ ! -f $CMSSW_BASE/o2o/replaces/${record}_${type}/${key}.txt ]
then
    echo "ERROR: replaces/${record}_${type}/${key}.txt does not exist.  Exiting."
    exit
fi

if [ -f $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py ]
    then
    echo "Payload for ${record} ${type} ${key} already written."
else
    echo "Creating $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py"
    cat $CMSSW_BASE/src/CondTools/L1Trigger/scripts/writePayloadStub.txt >& $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    perl -pi~ -e "s/CRUZET/${tagbase}/g" $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    rm $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py~

    echo "process.l1CSCTFConfig.ptLUT_path = '$CMSSW_BASE/o2o/PtLUT.dat'" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    echo "process.L1TriggerKeyDummy.objectKeys = cms.VPSet(cms.PSet(" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    echo "    record = cms.string('${record}')," >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    echo "    type = cms.string('${type}')," >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    echo "    key = cms.string('${key}')" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    echo "))" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py

    if [ ${sflag} -ne 0 ]
	then
	echo "Writing to sqlite_file:l1config.db instead of ORCON."
	echo "process.orcon.connect = cms.string('sqlite_file:l1config.db')" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
	echo "process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
	echo "process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')"  >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
	echo "process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')" >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    fi

    cat $CMSSW_BASE/o2o/replaces/${record}_${type}/${key}.txt >> $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py

    if [ ${nflag} -eq 0 ]
	then
	cmsRun $CMSSW_BASE/o2o/gen/${record}_${type}_${key}_payload.py
    fi
fi

exit
