#!/bin/bash

if [[ $2 == '' || $3 != '' ]]
then
  echo "This script accepts exactly 2 command line arguments"
  echo "Invoke it in this way:"
  echo "createTrendPlots.sh Path fileNameTemplate"
  echo "    Path:            name of the path where the files are"
  echo "    fileNameTemplate: prefix of the bad channel log file name before the run number (without _ )"
  echo "Exiting."
  exit 1
fi

getOfflineDQMDataGeneric.sh $1 QualityLog $2
moveTrendPlots.sh $1
