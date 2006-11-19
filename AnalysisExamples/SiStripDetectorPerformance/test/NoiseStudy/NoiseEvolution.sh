#!/bin/sh

path=$1

runs=`ls ${path}/Display_PedNoise_RunNb_*.root | sed -e "s@[^0-9]@@g" | sort -r`

arun=`echo $runs`

root -b -l -q "NoiseEvolution.C(\"$path\",\"$arun\")"
