#!/bin/bash

inputPath=/storage/data1/SiStrip/SiStripDQM/output/cruzet

eval `scramv1 runtime -sh`

mkdir log

for file in `ls $inputPath/*Alone.root`
do
  run=`echo $file| awk -F"R00" '{print $2}' | awk -F"_" '{print int($1)}'`

  echo $file $run

  cat template_SiStripQualityHotStripIdentifierRoot.cfg |sed -e "s@insertRun@$run@g"  -e "s@insertInputDQMfile@$
file@"  -e "s@@@" > log/SiStripQualityHotStripIdentifierRoot_$run.cfg

  cmsRun log/SiStripQualityHotStripIdentifierRoot_$run.cfg > log/SiStripQualityHotStripIdentifierRoot_$run.log


done
