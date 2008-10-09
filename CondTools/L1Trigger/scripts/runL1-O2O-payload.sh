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

if [ ! -f replaces/${record}_${type}_${key}_replaces.txt ]
then
    echo "ERROR: replaces/${record}_${type}_${key}_replaces.txt does not exist.  Exiting."
    exit
fi

if [ -f gen/${record}_${type}_${key}_payload.py ]
    then
    echo "Payload for ${record} ${type} ${key} already written."
else
    echo "Creating gen/${record}_${type}_${key}_payload.py"
    cat writePayloadStub.txt >& gen/${record}_${type}_${key}_payload.py

    echo "process.L1TriggerKeyDummy.subsystemKeys = cms.VPSet(cms.PSet(" >> gen/${record}_${type}_${key}_payload.py
    echo "    record = cms.string('${record}')," >> gen/${record}_${type}_${key}_payload.py
    echo "    type = cms.string('${type}')," >> gen/${record}_${type}_${key}_payload.py
    echo "    key = cms.string('${key}')" >> gen/${record}_${type}_${key}_payload.py
    echo "))" >> gen/${record}_${type}_${key}_payload.py

    if [ ${sflag} -ne 0 ]
	then
	echo "Writing to sqlite_file:l1config.db instead of ORCON."
	echo "process.orcon.connect = cms.string('sqlite_file:l1config.db')" >> gen/${record}_${type}_${key}_payload.py
	echo "process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')" >> gen/${record}_${type}_${key}_payload.py
	echo "process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')"  >> gen/${record}_${type}_${key}_payload.py
	echo "process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')" >> gen/${record}_${type}_${key}_payload.py
    fi

    cat replaces/${record}_${type}_${key}_replaces.txt >> gen/${record}_${type}_${key}_payload.py

    if [ ${nflag} -eq 0 ]
	then
	cmsRun gen/${record}_${type}_${key}_payload.py
    fi
fi

exit
