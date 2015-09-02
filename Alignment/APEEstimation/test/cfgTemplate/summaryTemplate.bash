#!/bin/bash


base="$CMSSW_BASE/src/Alignment/APEEstimation/test/cfgTemplate"



cmsRun $base/apeEstimatorSummary_cfg.py_THE_COMMANDS_


cmsRun $base/apeLocalSetting_cfg.py_THE_COMMANDS_



if [ $? -eq 0 ] ; then
  echo "\nAPE DB-Object created"
else
  echo "\nNo APE DB-Object created"
fi

if [ -a alignment.log ] ; then
  rm alignment.log
fi




