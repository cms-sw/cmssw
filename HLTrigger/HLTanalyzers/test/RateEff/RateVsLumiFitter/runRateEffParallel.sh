#!/bin/bash


### setup

# set location of RateEff cfg and log files
CFG_FILE_DIR=/1TB/rossin/Trigger/HighPU/cfgs
LOG_FILE_DIR=/1TB/rossin/Trigger/HighPU/logs

# set list of datasets (RateEff cfg file name stubs)
DS_LIST="
  AlphaT
  DoubleMu
  EleHadEG12
  HTMHT
  MET
  MuHad
  MultiJet
  PhotonDoubleEle
  PhotonDoublePhoton
  PhotonHad
  PhotonPhoton
  RMR
  SingleMu"

# set list of config file name prefixes (one per run)
CFG_FILE_PREFIX_LIST="
  hltmenu_HighPU_r179828_
  hltmenu_3E33_r178479_"

# set cfg file postfix (must be identical for all runs)
CFG_FILE_POSTFIX=_forHighPU_cfg.py


### Create log file directory.

echo "Creating log file directory $LOG_FILE_DIR"
mkdir -p $LOG_FILE_DIR


### Go to RateEff. Source environment.

RATE_EFF_DIR=$CMSSW_BASE/src/HLTrigger/HLTanalyzers/test/RateEff

echo "Changing into RateEff directory $RATE_EFF_DIR"
cd $RATE_EFF_DIR

echo "Sourcing RateEff environment"
source setup.sh


### process all cfg files in parallel

for DS in $DS_LIST
do
  echo "Processing 'dataset' $DS"
  for CFG_FILE_PREFIX in $CFG_FILE_PREFIX_LIST
  do
    CFG_FILE="${CFG_FILE_DIR}/${CFG_FILE_PREFIX}${DS}${CFG_FILE_POSTFIX}"
    LOG_FILE="${LOG_FILE_DIR}/${CFG_FILE_PREFIX}${DS}${CFG_FILE_POSTFIX}.log"
    echo "Starting in the background RateEff for $CFG_FILE"
    nohup ./OHltRateEff $CFG_FILE > $LOG_FILE &
  done
done


### Go to original directory.

echo "Changing back into original directory"
cd -


### Watch the processes and log files.

echo
echo "Will now watch RateEff processes"
sleep 3
watch -t "echo 'The following RateEff processes are still running (abort with Ctrl-c):' ; echo ; ps aux | grep OHltRateEff | grep -v grep"
