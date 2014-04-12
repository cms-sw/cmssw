#!/bin/bash

################################################
### CONFIGURATION
################################################

# set location of RateEff cfg and log files
DATA_DIR=/1TB/hartl/ratesVsLumiHighPileup
CFG_FILE_DIR=$DATA_DIR/cfgs
LOG_FILE_DIR=$DATA_DIR/logs
ROOT_FILE_DIR=$DATA_DIR/root

# set list of datasets (RateEff cfg file name stubs)

#DS_ARR=("AlphaT" "DoubleMu" "EleHadEG12" "HTMHT" "MET" "MuHad" "MultiJet" "PhotonDoubleEle" "PhotonDoublePhoton" "PhotonHad" "PhotonPhoton" "RMR" "SingleMu")

#DS_ARR=("Tau")

DS_ARR=("Jet")

# set list of config file name prefixes (one per run)
FILE_PREFIX_ARR=("hltmenu_HighPU_r179828_" "hltmenu_3E33_r178479_")

# set list of config file lumi scale factors (one per run)
LUMI_SCALE_FACTOR_ARR=("131.8" "1")

# set file postfixes
CFG_FILE_POSTFIX="_forHighPU.cfg"
LOG_FILE_POSTFIX="_forHighPU.log"
ROOT_FILE_POSTFIX="_forHighPU.root"

# set RateEff location
RATE_EFF_DIR=$CMSSW_BASE/src/HLTrigger/HLTanalyzers/test/RateEff

# set code snippet file
FILE_LIST_CODE_SNIPPET_FILE=$RATE_EFF_DIR/RateVsLumiFitter/file_list_code_snippet.icc


################################################
### EXECUTION
################################################

# Create log file directory.
echo "Creating log file directory $LOG_FILE_DIR"
mkdir -p $LOG_FILE_DIR


# Go to RateEff. Source environment.
echo "Changing into RateEff directory $RATE_EFF_DIR"
cd $RATE_EFF_DIR
echo "Sourcing RateEff environment"
source setup.sh

# Process all cfg files in parallel. Generate file list code snippet.
nbDS=${#DS_ARR[@]}
nbRuns=${#FILE_PREFIX_ARR[@]}

rm $FILE_LIST_CODE_SNIPPET_FILE
touch $FILE_LIST_CODE_SNIPPET_FILE

for (( i=0; i<${nbDS}; i++ ))
do
  DS=${DS_ARR[$i]}
  echo "Processing 'dataset' ${DS}"

  echo "  if (DS==\"${DS}\") {" >> $FILE_LIST_CODE_SNIPPET_FILE

  for (( j=0; j<${nbRuns}; j++ ))
  do
    FILE_PREFIX=${FILE_PREFIX_ARR[$j]}
    CFG_FILE="${CFG_FILE_DIR}/${FILE_PREFIX}${DS}${CFG_FILE_POSTFIX}"
    LOG_FILE="${LOG_FILE_DIR}/${FILE_PREFIX}${DS}${LOG_FILE_POSTFIX}"
    echo "Starting RateEff in the background for $CFG_FILE"
    nohup ./OHltRateEff $CFG_FILE > $LOG_FILE &

    ROOT_FILE="${ROOT_FILE_DIR}/${FILE_PREFIX}${DS}${ROOT_FILE_POSTFIX}"
    LUMI_SCALE_FACTOR=${LUMI_SCALE_FACTOR_ARR[$j]}
    echo "    f[fileCounter] = new TFile(\"$ROOT_FILE\"); vlumiSFperFile[fileCounter++]=$LUMI_SCALE_FACTOR;" >> $FILE_LIST_CODE_SNIPPET_FILE
  done

  echo '  };' >> $FILE_LIST_CODE_SNIPPET_FILE
done

echo "Produced file list code snippet for rate fitter macro: $FILE_LIST_CODE_SNIPPET_FILE"

# Go to original directory.
echo "Changing back into original directory"
cd -

# Watch the processes and log files.
echo
echo "Will now watch RateEff processes"
sleep 3
watch -t "echo 'The following RateEff processes are still running (abort with Ctrl-c):' ; echo ; ps aux | grep OHltRateEff | grep -v grep"
