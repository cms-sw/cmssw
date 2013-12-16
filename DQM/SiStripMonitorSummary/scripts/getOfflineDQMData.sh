#!/bin/bash

if [[ $4 == '' || $5 != '' ]]
then
  echo "This script accepts exactly 4 command line arguments"
  echo "Invoke it in this way:"
  echo "getOfflineDQMData.sh DBName DBAccount MonitoredVariable DBTag"
  echo "    DBname:            name of the database (Ex: cms_orcoff_prod)"
  echo "    DBAccount:         name of the database account (Ex: CMS_COND_31X_STRIP)"
  echo "    monitoredVariable: must be one among SiStripBadChannel, SiStripFedCabling, SiStripVoltage or RunInfo"
  echo "    DBTag:             name of the database tag (Ex: SiStripBadComponents_OfflineAnalysis_GR09_31X_v1_offline)"
  echo "Exiting."
  exit 1
fi

# The script accepts 4 command line parameters:
# Example: cms_orcoff_prod
export DBName=$1
# Example: CMS_COND_31X_STRIP
export DBAccount=$2
# Example: SiStripBadChannel
export monitoredVariable=$3
# Example: SiStripBadComponents_OfflineAnalysis_GR09_31X_v1_offline
export DBTag=$4

if [[ $monitoredVariable == "SiStripBadChannel" ]]
then
  baseDir=QualityLog
  baseName=QualityInfo_Run
elif [[ $monitoredVariable == "SiStripFedCabling" ]]
then
  baseDir=CablingLog
  baseName=QualityInfoFromCabling_Run
elif [[ $monitoredVariable == "SiStripVoltage" ]]
then
  baseDir=QualityLog
  baseName=QualityInfo_Run
elif [[ $monitoredVariable == "RunInfo" ]]
then
  baseDir=QualityLog
  baseName=QualityInfo_Run
else
  echo "The monitored variable that was entered is not valid!"
  echo "Valid choices are: SiStrip, SiStripFedCabling, SiStripVoltage or RunInfo."
  echo "Exiting."
  exit 1
fi

getOfflineDQMDataGeneric.sh /afs/cern.ch/cms/tracker/sistrcalib/WWW/CondDBMonitoring/$DBName/$DBAccount/DBTagCollection/$monitoredVariable/$DBTag $baseDir $baseName

