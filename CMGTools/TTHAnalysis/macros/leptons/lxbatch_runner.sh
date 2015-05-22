#!/bin/bash
export SCRAM_ARCH=slc5_amd64_gcc462
WORK=$1; shift
SRC=$1; shift
cd $SRC; 
eval $(scramv1 runtime -sh);
cd $WORK;
exec $*
