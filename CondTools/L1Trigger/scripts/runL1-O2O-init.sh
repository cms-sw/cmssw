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
      h) echo "Usage: [-s] [-n]"
          echo "  -s: write to sqlite file instead of ORCON"
          echo "  -n: do not run init.py"
          exit
          ;;
  esac
done
shift $(($OPTIND - 1))

tagbase=$1

if [ ${sflag} -ne 0 ]
    then
    echo "Setting up sqlite_file:l1config.db"
else
    echo "Setting up ORCON CMS_COND_21X_L1T account"
fi

if [ ${nflag} -eq 0 ]
    then    
    if [ ! -d $CMSSW_BASE/o2o/gen ]
	then
	mkdir $CMSSW_BASE/o2o/gen
    fi

    # setup schema
    if [ ${sflag} -ne 0 ]
	then
	cmscond_bootstrap_detector -D L1T -f $CMSSW_BASE/o2o/dbconfigSqlite.xml -b $CMSSW_BASE
    else
	cmscond_bootstrap_detector -D L1T -f $CMSSW_BASE/o2o/dbconfigORCON.xml -b $CMSSW_BASE
    fi

    # generate .py
    cp $CMSSW_BASE/src/CondTools/L1Trigger/test/init_cfg.py $CMSSW_BASE/o2o/gen/init_cfg.py
    echo "process.L1CondDBPayloadWriter.L1TriggerKeyListTag = cms.string('L1TriggerKeyList_${tagbase}_offline')" >> $CMSSW_BASE/o2o/gen/init_cfg.py

    if [ ${sflag} -ne 0 ]
	then
	echo "process.L1CondDBPayloadWriter.offlineDB = cms.string('sqlite_file:l1config.db')" >> $CMSSW_BASE/o2o/gen/init_cfg.py
	echo "process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('.')" >> $CMSSW_BASE/o2o/gen/init_cfg.py
    else
	echo "process.L1CondDBPayloadWriter.offlineDB = cms.string('oracle://cms_orcon_prod/CMS_COND_21X_L1T')" >> $CMSSW_BASE/o2o/gen/init_cfg.py
	echo "process.L1CondDBPayloadWriter.offlineAuthentication = cms.string('/nfshome0/xiezhen/conddb')" >> $CMSSW_BASE/o2o/gen/init_cfg.py
    fi

    # write empty L1TriggerKeyList
    cmsRun $CMSSW_BASE/o2o/gen/init_cfg.py
fi

exit
