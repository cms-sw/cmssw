#! /bin/bash

# Exit the script immediately if any command fails
set -e

# Enable pipefail to propagate the exit status of the entire pipeline
set -o pipefail

FOLDER_FILES="/data/user/${USER}/"
DATASET="/RelValTTbar_14TeV/CMSSW_14_1_0_pre6-PU_141X_mcRun4_realistic_v1_STD_2026D110_PU-v3/GEN-SIM-DIGI-RAW"
FILES=( $(dasgoclient -query="file dataset=${DATASET}" --limit=-1 | sort | head -4) )

for f in ${FILES[@]}; do
  # Create full MYPATH if it does not exist
  MYPATH=$(dirname ${f})
  if [ ! -d "${FOLDER_FILES}${MYPATH}" ]; then
    echo "mkdir -p ${FOLDER_FILES}${MYPATH}"
    mkdir -p ${FOLDER_FILES}${MYPATH}
  fi
  if [ -e "/eos/cms/${f}" ]; then
    if [ ! -e "${FOLDER_FILES}${f}" ]; then
      echo "cp /eos/cms/$f ${FOLDER_FILES}${MYPATH}"
      cp /eos/cms/$f ${FOLDER_FILES}${MYPATH}
    fi
  fi
done

LOCALPATH=${FOLDER_FILES}$(dirname ${FILES[0]})
echo "Local repository: |${LOCALPATH}|"
LOCALFILES=$(ls -1 ${LOCALPATH})
ALL_FILES=""
for f in ${LOCALFILES[@]}; do
  ALL_FILES+="file:${LOCALPATH}/${f},"
done
# Remove the last character
ALL_FILES="${ALL_FILES%?}"
echo "Discovered files: $ALL_FILES"

cmsDriver.py Phase2 -s L1P2GT,HLT:75e33_timing --processName=HLTX \
  --conditions auto:phase2_realistic_T33 --geometry ExtendedRun4D110 \
  --era Phase2C17I13M9 \
  --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
  --eventcontent FEVTDEBUGHLT \
  --filein=${ALL_FILES} \
  --mc --nThreads 4 --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
  -n 1000 --no_exec --output={}

if [ -e 'Phase2_L1P2GT_HLT.py' ]; then
  if [ ! -d 'patatrack-scripts' ]; then
    git clone https://github.com/cms-patatrack/patatrack-scripts --depth 1
  fi
  patatrack-scripts/benchmark -j 8 -t 16 -s 16 -e 1000 --no-run-io-benchmark --event-skip 100 --event-resolution 10 -k Phase2Timing_resources.json -- Phase2_L1P2GT_HLT.py
  mergeResourcesJson.py logs/step*/pid*/Phase2Timing_resources.json >Phase2Timing_resources.json
  if [ -e "$(dirname $0)/augmentResources.py" ]; then
    python3 $(dirname $0)/augmentResources.py
  fi
fi

cmsDriver.py NGTScouting -s L1P2GT,HLT:NGTScouting --processName=NLTX \
  --conditions auto:phase2_realistic_T33 --geometry ExtendedRun4D110 \
  --era Phase2C17I13M9 \
  --customise SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000 \
  --eventcontent FEVTDEBUGHLT \
  --filein=${ALL_FILES} \
  --mc --nThreads 4 --inputCommands='keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
  -n 1000 --no_exec --output={}

if [ -e 'NGTScouting_L1P2GT_HLT.py' ]; then
  if [ ! -d 'patatrack-scripts' ]; then
    git clone https://github.com/cms-patatrack/patatrack-scripts --depth 1
  fi
  patatrack-scripts/benchmark -j 8 -t 16 -s 16 -e 1000 --no-run-io-benchmark --event-skip 100 --event-resolution 10 -k Phase2Timing_resources.json -- NGTScouting_L1P2GT_HLT.py
  mergeResourcesJson.py logs/step*/pid*/Phase2Timing_resources.json >Phase2Timing_resources_NGT.json
fi
