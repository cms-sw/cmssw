#!/bin/bash
WORK=$PWD
SRC=$CMSSW_BASE/src
QUEUE=8nh
OPTS=""
if echo "X$1" | grep -q '^X-[a-zA-Z0-9]'; then 
    QUEUE=$(echo "X$1" | sed 's/^X-//');
    shift;
    while echo "X$1" | grep -q '^X-[a-zA-Z0-9]'; do
        OPTS="$OPTS $1 $2";
        shift; shift;
    done;
fi;
bsub -q $QUEUE $OPTS $WORK/lxbatch_runner.sh $WORK $SRC $*
