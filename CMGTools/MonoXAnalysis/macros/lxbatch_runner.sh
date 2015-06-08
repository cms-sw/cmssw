#!/bin/bash
export SCRAM_ARCH=slc6_amd64_gcc481
WORK=$1; shift
SRC=$1; shift
cd $SRC; 
eval $(scramv1 runtime -sh);
cd $WORK;
exec $*
