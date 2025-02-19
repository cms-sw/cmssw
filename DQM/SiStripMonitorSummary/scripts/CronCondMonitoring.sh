#!/bin/sh

#Autor subir.sarkar@cern.ch

set -o nounset

echo "====> Starting CondDB cron job at $(date)<===="

DEBUG=1
PROGNAME=/home/cmstacuser/CMSSWReleasesForCondDB/CMSSW_2_1_12/src/RunCondDBOffline21X_PedNoise.sh  
#PROGNAME=/bin/ls  
COMMAND=cmsRun
let "count = $(ps --no-headers -l -C $COMMAND | wc -l)"
if [ "$count" -gt 0 ]; then
 cmdrunning=$(ps --no-headers -C $COMMAND -o cmd)
 if echo $cmdrunning | grep "log21X/MainCfg" > /dev/null
 then
   if [ $DEBUG -gt 0 ]; then
     echo Time: $(date '+%D-%T') Caller: $(basename $0) - $COMMAND already running ...
     ps --columns 180 --no-headers -fl -C $COMMAND | grep "log21X/Main"
   fi
   echo "====> EXIT <===="
   exit 1
 fi
fi

$PROGNAME
status=$?
echo -- Finished at $(date) with status=$status --
exit $status
echo ----
