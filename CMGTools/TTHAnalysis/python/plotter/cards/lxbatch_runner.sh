#!/bin/bash
WORK=$1; shift
SRC=$1; shift
cd $SRC; 
eval $(scramv1 runtime -sh);
cd $WORK;
export COMBINE_NO_LOGFILES=0
bash $*
