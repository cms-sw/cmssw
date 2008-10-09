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
      h) echo "Usage: [-s] [-n] tsckey runnum"
          echo "  -s: write to sqlite file instead of ORCON"
          echo "  -n: generate .py file but do not run it"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

tsckey=$1
runnum=$2

if [ ! -f gen/${tsckey}_${runnum}_iov.py ]
then
    echo "Creating gen/${tsckey}_${runnum}_iov.py"
    cat writeIOVStub.txt >& gen/${tsckey}_${runnum}_iov.py

    echo "process.L1TriggerKeyDummy.tscKey = cms.string('${tsckey}')" >> gen/${tsckey}_${runnum}_iov.py
    echo "process.EmptyIOVSource.firstRun = cms.untracked.uint32(${runnum})" >> gen/${tsckey}_${runnum}_iov.py
    echo "process.EmptyIOVSource.lastRun = cms.untracked.uint32(${runnum})" >> gen/${tsckey}_${runnum}_iov.py

    if [ ${sflag} -ne 0 ]
        then
        echo "Writing to sqlite_file:l1config.db instead of ORCON."
	echo "process.orcon.connect = cms.string('sqlite_file:l1config.db')" >> gen/${tsckey}_${runnum}_iov.py
	echo "process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')" >> gen/${tsckey}_${runnum}_iov.py
	echo "process.L1CondDBIOVWriter.offlineDB = cms.string('sqlite_file:l1config.db')" >> gen/${tsckey}_${runnum}_iov.py
	echo "process.L1CondDBIOVWriter.offlineAuthentication = cms.string('.')" >> gen/${tsckey}_${runnum}_iov.py
    fi
fi

if [ ${nflag} -eq 0 ]
    then
    cmsRun gen/${tsckey}_${runnum}_iov.py
fi

exit
