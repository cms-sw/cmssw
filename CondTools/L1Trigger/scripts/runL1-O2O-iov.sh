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

runnum=$1
tsckey=$2
tagbase=$3

if [ ! -f $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py ]
then
    echo "Creating $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py"
    cat $CMSSW_BASE/src/CondTools/L1Trigger/scripts/writeIOVStub.txt >& $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
    perl -pi~ -e "s/CRUZET/${tagbase}/g" $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
    rm $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py~

    echo "process.L1CondDBIOVWriter.tscKey = cms.string('${tsckey}')" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
    echo "process.EmptyIOVSource.firstValue = cms.uint64(${runnum})" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
    echo "process.EmptyIOVSource.lastValue = cms.uint64(${runnum})" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py

    if [ ${sflag} -ne 0 ]
        then
        echo "Writing to sqlite_file:l1config.db instead of ORCON."
	echo "process.orcon.connect = cms.string('sqlite_file:l1config.db')" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
	echo "process.orcon.DBParameters.authenticationPath = cms.untracked.string('.')" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
	echo "process.L1CondDBIOVWriter.offlineDB = cms.string('sqlite_file:l1config.db')" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
	echo "process.L1CondDBIOVWriter.offlineAuthentication = cms.string('.')" >> $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
    fi
fi

if [ ${nflag} -eq 0 ]
    then
    cmsRun $CMSSW_BASE/o2o/gen/${tsckey}_${runnum}_iov.py
fi

exit
