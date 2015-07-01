#!/bin/bash

WHAT=$1; if [[ "$1" == "74X" ]]; then echo "synch.sh <what>"; exit 1; fi

if [[ "$HOSTNAME" == "cmsphys06" ]]; then
    T="/data1/emanuele/monox/TREES_SYNCH_741_ONEFILE/";
    J=6;
else
    T="/cmshome/dimarcoe/TREES_SYNCH_741_ONEFILE/";
    J=6;
fi
COREOPT="-P $T --s2v -j $J -l 5.0"
COREY="mcAnalysis.py ${COREOPT} -G  "
FEV=" -F mjvars/t \"$T/0_eventvars_mj_v1/evVarFriend_{cname}.root\" "

ROOT="plots/050515/v1.0/$WHAT"

RUNY="${COREY} mca-74X.txt --s2v -u "
RUNYSR="${RUNY} sync/monojet_twiki.txt "
RUNY2M="${RUNY} sync/zmumu_twiki.txt "
RUNY1M="${RUNY} sync/wmunu_twiki.txt "
RUNY1G="${RUNY} sync/gjets_twiki.txt "

case $WHAT in
sr)
        echo "python ${RUNYSR} $FEV --xp TTbar --xp DYLL --xp WJets --xp RSGravGaGa "
;;
wmunu)
        echo "python ${RUNY1M} $FEV --sp WJets --xp ADD --xp TTLep --xp TTbar --xp DYLL --xp RSGravGaGa "
;;
zmumu)
        echo "python ${RUNY2M} $FEV --sp DYLL --xp ADD --xp TTLep --xp TTbar --xp WJets --xp RSGravGaGa "
;;
gjets)
        echo "python ${RUNY1G} $FEV --sp DYLL --xp ADD --xp TTLep --xp TTbar --xp WJets --xp DYLL "
esac;
