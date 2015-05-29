#!/bin/bash

if [[ "$HOSTNAME" == "cmsphys06" ]]; then
    T="/data1/emanuele/monox/TREES_150515_MET200SKIM";
    J=6;
else
    T="/cmshome/dimarcoe/TREES_060515_MET200SKIM";
    J=6;
fi

ASIMOV=""
if echo "$1" | grep -q "asimov"; then ASIMOV="$1"; shift; fi

OPTIONS=" -P $T --s2v -j $J -l 5.0 -f "
SYSTS="systsEnv.txt"

if [[ $ASIMOV == "asimov" ]] ; then
    OPTIONS="${OPTIONS} --asimov "
fi

if [[ "$1" == "" ]] ; then
    X="monojet"
    OPTIONS="${OPTIONS} -F mjvars/t \"$T/0_eventvars_mj_v1/evVarFriend_{cname}.root\" "
    OPT_SHAPE="${OPTIONS} --od cards/shape "
    OPT_COUNT="${OPTIONS} -A \"lep veto\" \"met500\" \"metNoMu_pt>500\" --od cards/counting "
    echo "python makeShapeCards.py mca-Phys14.txt sr/monojet.txt  'metNoMu_pt' '[200,250,300,350,400,500,650,1000]' $SYSTS $OPT_SHAPE -o ${X} "
    echo "python makeCountingCards.py mca-Phys14.txt sr/monojet.txt $SYSTS $OPT_COUNT -o ${X} "
    echo "Done at $(date)"
fi;
