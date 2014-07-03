#!/bin/bash
# generic script for running sources



if [ $# -ne 2 ]; then
    echo "need two arguments: subsystem and cfg string"
    exit 1
fi
me=$1
cfg=$2

echo "$0: starting loop of ${me} for ${cfg}"

cd $HOME/$CMSSW_VERSION/src/DQM/Integration/python/test
eval `scramv1 runtime -sh` > /dev/null

CFG=${me}_dqm_sourceclient-${cfg}_cfg.py
if [ ! -f $CFG  ]; then
    echo "Can't find cfg file $CFG in $PWD; exiting."
    exit 2
fi

LOGDIR=/tmp/${USER}

[ -d $LOGDIR ] || mkdir -p $LOGDIR 

LOGFILE=${LOGDIR}/${me}_${cfg}.out
# keep one old logfile
[ -f $LOGFILE ] && mv $LOGFILE ${LOGFILE}.old

# redirect stderr and stdout to log file
exec 1>$LOGFILE 2>&1
# close stdin
exec 0<&-


while ( true ) ; 
  do 
  date
  echo "---------------------> Starting CMSRUN " 
  (set -x; cmsRun ${me}_dqm_sourceclient-${cfg}_cfg.py; sleep 10 )
done


