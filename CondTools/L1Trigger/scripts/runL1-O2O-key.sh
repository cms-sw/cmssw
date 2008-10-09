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
      h) echo "Usage: [-s] [-n] tsckey"
	  echo "  -s: write to sqlite file instead of ORCON"
	  echo "  -n: generate .py file but do not run it"
	  exit
	  ;;
  esac
done
shift $(($OPTIND - 1))

tsckey=$1

if [ ! -f gen/${tsckey}_key.py ]
then
    echo "Creating gen/${tsckey}_key.py"
    cat writeKeyStub.txt >& gen/${tsckey}_key.py

    echo "process.L1TriggerKeyOnline.tscKey = cms.string('${tsckey}')" >> gen/${tsckey}_key.py

    if [ ${sflag} -ne 0 ]
        then
        echo "Writing to sqlite_file:l1config.db instead of ORCON."
	echo "process.orcon.connect = cms.string('sqlite_file:l1config.db')" >> gen/${tsckey}_key.py
	echo "process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')" >> gen/${tsckey}_key.py
	echo "process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')" >> gen/${tsckey}_key.py
	echo "process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')" >> gen/${tsckey}_key.py
    fi
fi

if [ ${nflag} -eq 0 ]
    then
    cmsRun gen/${tsckey}_key.py
fi

exit
