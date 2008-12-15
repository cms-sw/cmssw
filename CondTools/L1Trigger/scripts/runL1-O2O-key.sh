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
tagbase=$2
shift 2

# Remaining arguments are records to include

if [ ! -f $CMSSW_BASE/o2o/gen/${tsckey}_key.py ]
then
    echo "Creating $CMSSW_BASE/o2o/gen/${tsckey}_key.py"
    cat $CMSSW_BASE/src/CondTools/L1Trigger/scripts/writeKeyStub.txt >& $CMSSW_BASE/o2o/gen/${tsckey}_key.py
    perl -pi~ -e "s/CRUZET/${tagbase}/g" $CMSSW_BASE/o2o/gen/${tsckey}_key.py
    rm $CMSSW_BASE/o2o/gen/${tsckey}_key.py~

    echo "process.L1SubsystemKeysOnline.tscKey = cms.string('${tsckey}')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py
    echo "process.L1SubsystemKeysOnline.onlineAuthentication = cms.string('${CMSSW_BASE}/o2o')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py
    echo "process.L1RCTObjectKeysOnline.onlineAuthentication = cms.string('${CMSSW_BASE}/o2o')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py

    if [ ${sflag} -ne 0 ]
        then
        echo "Writing to sqlite_file:l1config.db instead of ORCON."
	echo "process.orcon.connect = cms.string('sqlite_file:l1config.db')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py
	echo "process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py
	echo "process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py
	echo "process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')" >> $CMSSW_BASE/o2o/gen/${tsckey}_key.py
    fi
fi

if [ ${nflag} -eq 0 ]
    then
    cmsRun $CMSSW_BASE/o2o/gen/${tsckey}_key.py
fi

exit
