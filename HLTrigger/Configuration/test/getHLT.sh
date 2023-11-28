#!/bin/bash

# ConfDB directory hosting the HLT configurations
CONFDBDIR="/dev/CMSSW_13_3_0"

# ConfDB configurations to use
#  - if no explicit version, the most recent one is taken
#  - to use explicit version, specify it in the entries below
#  - to skip a given configuration, remove or comment the corresponding entry in the array TABLES
#  - new configurations can be added by expanding the array TABLES
#  - for new configurations, ensure that the corresponding "auto" GTs are defined in
#     Configuration/HLT/python/autoCondHLT.py , and
#     HLTrigger/Configuration/python/Tools/options.py
declare -A TABLES=(
  ["FULL"]="${CONFDBDIR}/HLT"
  ["GRun"]="${CONFDBDIR}/GRun"
  ["HIon"]="${CONFDBDIR}/HIon"
  ["PIon"]="${CONFDBDIR}/PIon"
  ["PRef"]="${CONFDBDIR}/PRef"
)

# command-line arguments
VERBOSE=false # print extra messages to stdout
DBPROXYOPTS="" # db-proxy configuration
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v) VERBOSE=true; shift;;
    --dbproxy) DBPROXYOPTS="${DBPROXYOPTS} --dbproxy"; shift;;
    --dbproxyhost) DBPROXYOPTS="${DBPROXYOPTS} --dbproxyhost $2"; shift; shift;;
    --dbproxyport) DBPROXYOPTS="${DBPROXYOPTS} --dbproxyport $2"; shift; shift;;
    *) shift;;
  esac
done

# remove spurious whitespaces and tabs from DBPROXYOPTS
DBPROXYOPTS=$(echo "${DBPROXYOPTS}" | xargs)

# log: print to stdout only if VERBOSE=true
function log() {
  $VERBOSE && echo -e "$@"
}

# path to directory hosting this script
TESTDIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# ensure that directory hosting this script corresponds to ${CMSSW_BASE}/src/HLTrigger/Configuration/test
if [ "${TESTDIR}" != "${CMSSW_BASE}"/src/HLTrigger/Configuration/test ]; then
  printf "\n%s\n" "ERROR -- the directory hosting getHLT.sh [1] does not correspond to \${CMSSW_BASE}/src/HLTrigger/Configuration/test [2]"
  printf "%s\n"   "         [1] ${TESTDIR}"
  printf "%s\n\n" "         [2] ${CMSSW_BASE}/src/HLTrigger/Configuration/test"
  exit 1
fi

# ensure that the python/ directory hosting cff fragments exists
if [ ! -d "${CMSSW_BASE}"/src/HLTrigger/Configuration/python ]; then
  printf "\n%s\n" "ERROR -- the directory \${CMSSW_BASE}/src/HLTrigger/Configuration/python [1] does not exist"
  printf "%s\n\n" "         [1] ${CMSSW_BASE}/src/HLTrigger/Configuration/python"
  exit 1
fi

INITDIR="${PWD}"

# execute the ensuing steps from ${CMSSW_BASE}/src/HLTrigger/Configuration/test
cd "${CMSSW_BASE}"/src/HLTrigger/Configuration/test

# create cff fragments and cfg configs
for TABLE in "${!TABLES[@]}"; do
  CONFIG="${TABLES[${TABLE}]}"
  echo "${TABLE} (config: ${CONFIG})"

  # cff fragment of each HLT menu (do not use any conditions or L1T override)
  log "  creating cff fragment of HLT menu..."
  hltGetConfiguration "${CONFIG}" --cff --data --type "${TABLE}" ${DBPROXYOPTS} > ../python/HLT_"${TABLE}"_cff.py

  # cff fragment of EventContents (only for FULL config)
  if [ "${TABLE}" = "FULL" ]; then
    log "  creating cff fragment of EventContents..."
    ./getEventContent.py "${CONFIG}" ${DBPROXYOPTS} > ../python/HLTrigger_EventContent_cff.py
  fi

  # cff fragment of PrimaryDatasets of each HLT menu (except for FULL config)
  if [ "${TABLE}" != "FULL" ]; then
    log "  creating cff fragment of Primary Datasets..."
    ./getDatasets.py "${CONFIG}" ${DBPROXYOPTS} > ../python/HLTrigger_Datasets_"${TABLE}"_cff.py
  fi

  # GlobalTag
  AUTOGT="auto:run3_hlt_${TABLE}"
  if [ "${TABLE}" = "Fake1" ] || [ "${TABLE}" = "Fake2" ] || [ "${TABLE}" = "2018" ]; then
    AUTOGT="auto:run2_hlt_${TABLE}"
  elif [ "${TABLE}" = "Fake" ]; then
    AUTOGT="auto:run1_hlt_${TABLE}"
  fi

  # standalone cfg file of each HLT menu (incl. FULL config)
  log "  creating full cfg of HLT menu..."
  hltGetConfiguration "${CONFIG}" --full --data --type "${TABLE}" --unprescale --process "HLT${TABLE}" --globaltag "${AUTOGT}" \
    --input "file:RelVal_Raw_${TABLE}_DATA.root" ${DBPROXYOPTS} > OnLine_HLT_"${TABLE}".py
done

cd "${INITDIR}"
