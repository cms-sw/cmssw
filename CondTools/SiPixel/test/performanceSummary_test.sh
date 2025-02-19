#!/bin/sh

USAGE="`basename $0` -write / -read"
case $# in
1)
  case "$1" in
        -write)  MODE="write"; ;;
        -read)   MODE="read";  ;;
        *)       echo $USAGE; exit 1; ;;
  esac
  shift
	;;
*)
	echo $USAGE; exit 1;
	;;
esac

eval `scramv1 runtime -sh`
SealPluginRefresh
export CORAL_AUTH_USER=""
export CORAL_AUTH_PASSWORD=""

if [ "$MODE" == "write" ] 
then
  rm sipixelperformancesummary.db
  rm sipixelperformancesummary.xml
  cmsRun performanceSummary_write.cfg

elif [ "$MODE" == "read" ]
then
  cmsRun performanceSummary_read.cfg

fi
