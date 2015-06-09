#!/bin/bash

WHAT=$1; if [[ "$1" == "" ]]; then echo "sync.sh <what>"; exit 1; fi

if [[ "$HOSTNAME" == "cmsphys06" ]]; then
    T="/data1/emanuele/monox/TREES_SYNCH_741";
    J=6;
else
    T="/cmshome/dimarcoe/TREES_SYNCH_741";
    J=6;
fi
COREOPT="-P $T --s2v -j $J -l 5.0"
COREY="mcAnalysis.py ${COREOPT} -G  "
FEV=" -F mjvars/t \"$T/0_eventvars_mj_v1/evVarFriend_{cname}.root\" "

ROOT="plots/050515/v1.0/$WHAT"

RUNY="${COREY} mca-74X.txt --s2v -u "
RUNYSR="${RUNY} sync/monojet.txt "

case $WHAT in
sr)
        #SF="-W 'puWeight'"
        SF=" "
        echo "python ${RUNYSR} $FEV $SF "
esac;
