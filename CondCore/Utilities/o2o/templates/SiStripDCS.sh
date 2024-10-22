#!/bin/sh
#
# wrapper script to run DCS O2O
# Last updated: Apr 21, 2017
# Author: Huilin Qu
#
# Usage: SiStripDCS.sh JOB_NAME

JOBNAME=$1

O2O_HOME=@root

source $O2O_HOME/scripts/setStrip.sh $JOBNAME
o2oRun_SiStripDCS.py $JOBNAME |tee -a $LOGFILE

# Exit with status of last command
# Make sure the last command is o2oRun_SiStripDCS.py!
exit $?
