#!/bin/bash
VERB=0; if [[ "$1" == "-v" ]]; then VERB=1; shift; fi;
WHAT=$1; shift; 
while [[ "$1" != "" ]]; do
    NAM=$(echo $1 | sed -e s/comb_*// -e s/.root//   | tr '[a-z]' '[A-Z]')
    if [[ "$NAM" == "COMB" ]]; then NAM=""; fi;
    FILES=$(ls -l higgsCombine${NAM}_${WHAT}.*.root 2> /dev/null | awk '{if ($5 > 1000) print}' | wc -l);
    if [[ "$FILES" == "0" ]]; then shift; continue; fi;
    if [[ "$VERB" == "1" ]]; then 
        ./hadd2 -f higgsCombine${NAM}_${WHAT}.root higgsCombine${NAM}_${WHAT}.*.root
    else
        ./hadd2 -f higgsCombine${NAM}_${WHAT}.root higgsCombine${NAM}_${WHAT}.*.root > /dev/null 2>&1
    fi;
    echo "higgsCombine${NAM}_${WHAT}.root   ($FILES files)"
    shift;
done
