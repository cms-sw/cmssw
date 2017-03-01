#!/bin/bash


declare -i counter=1

for file in $CMSSW_BASE/src/Alignment/APEEstimation/test/batch/workingArea/*.tcsh;

do

  bsub -J job${counter} -q cmscaf1nd -R "rusage[pool=3000]" "type=SLC5_64" tcsh $file
  
  counter=$counter+1
  
done

