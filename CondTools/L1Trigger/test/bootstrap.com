#!/bin/sh

lflag=0
while getopts 'lh' OPTION
do
  case $OPTION in
      l) lflag=1
          ;;
      h) echo "Usage: [-l]"
          echo "  -l: use mapping xml files in local CMSSW area"
          exit
          ;;
  esac
done

if [ ${lflag} -eq 0 ]
    then
	echo "Setting up sqlite_file:l1config.db with xml files from $CMSSW_RELEASE_BASE"
	cmscond_bootstrap_detector -D L1T -f $CMSSW_RELEASE_BASE/src/CondTools/L1Trigger/test/dbconfiguration.xml -b $CMSSW_RELEASE_BASE
	else
	echo "Setting up sqlite_file:l1config.db with xml files from $CMSSW_BASE"
	cmscond_bootstrap_detector -D L1T -f $CMSSW_BASE/src/CondTools/L1Trigger/test/dbconfiguration.xml -b $CMSSW_BASE
fi

exit
