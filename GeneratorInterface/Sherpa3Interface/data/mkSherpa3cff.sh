#!/bin/bash
#
#  file:        mkSherpa3cff.sh
#  description: Generate the CMSSW python fragment (cff) for an existing
#               Sherpa3 gridpack produced by mkSherpa3Gridpack.sh.
#               The runcard (Sherpa.yaml) is taken from inside the gridpack,
#               and the gridpack md5 checksum is included.
#
#  usage:       mkSherpa3cff.sh -gp <gridpack.tar.xz> [options]
#
#  options: -gp,  --gridpack path  the Sherpa3 gridpack tarball
#           -yml, --sh_yml   path  runcard in YAML format (default: Sherpa.yaml
#                                  extracted from the gridpack)
#           -w,   --weights  path  file with the Sherpa weight names, one per
#                                  line, as written by mkSherpa3Gridpack.sh
#                                  (Sherpa3_<origin_name>_weights.txt). Without
#                                  it the cff gets no SherpaWeightsBlock and the
#                                  event weights end up in the GEN output
#                                  unnamed. This script does not run Sherpa, so
#                                  the names cannot be determined here.
#                                  The SherpaWeightsBlock also contains
#                                  SherpaRenamedWeights /
#                                  SherpaRenamedVariationWeights: the
#                                  EDM-facing names ('+' -> 'p', '-' -> 'm')
#                                  the hadronizer stores, one per entry of the
#                                  matching Sherpa list, in the same order.
#           -o,   --output   path  output cff file name
#                                  (default: Sherpa3_<origin_name>_cff.py)
#           -h,   --help            display this help and exit
#
#  output:      Sherpa3_<origin_name>_cff.py with a Sherpa3GeneratorFilter
#               (Sherpa 3 always produces HepMC3).

set -uo pipefail

# +-----------------------------------------------------------------------------------------------+
# defaults & argument parsing
# +-----------------------------------------------------------------------------------------------+

GRIDPACK=""
SH_YML=""
CFFFILE=""
WEIGHTS_FILE=""

# print the header comment block (everything after the shebang up to the first
# non-comment line), so the help never goes stale when the header grows
print_help() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -gp|--gridpack) GRIDPACK="$2";  shift 2 ;;
    -yml|--sh_yml)  SH_YML="$2";    shift 2 ;;
    -w|--weights)   WEIGHTS_FILE="$2"; shift 2 ;;
    -o|--output)    CFFFILE="$2";   shift 2 ;;
    -h|--help)      print_help; exit 0 ;;
    *) echo "mkSherpa3cff: unknown option '$1' (use -h for help)"; exit 1 ;;
  esac
done

# +-----------------------------------------------------------------------------------------------+
# sanity checks
# +-----------------------------------------------------------------------------------------------+

if [ -z "${GRIDPACK}" ]; then
  echo "mkSherpa3cff: no gridpack given, use -gp/--gridpack (use -h for help)"
  exit 1
fi
if [ ! -f "${GRIDPACK}" ]; then
  echo "mkSherpa3cff: gridpack '${GRIDPACK}' not found"
  exit 1
fi
GRIDPACK="$(cd "$(dirname "${GRIDPACK}")" && pwd)/$(basename "${GRIDPACK}")"

# process/origin name from the gridpack file name:
# Sherpa3_<origin>_<SCRAM_ARCH>_<CMSSW_VERSION>_tarball.tar.xz
ORIGIN_NAME=$(basename "${GRIDPACK}")
ORIGIN_NAME="${ORIGIN_NAME%_tarball.tar.xz}"   # strip _tarball.tar.xz
ORIGIN_NAME="${ORIGIN_NAME#Sherpa3_}"          # strip Sherpa3_ prefix
ORIGIN_NAME=$(echo "${ORIGIN_NAME}" | sed -e 's/_[a-z0-9]*_amd64_gcc[0-9]*_.*$//')  # strip _<arch>_<release>

if [ -z "${CFFFILE}" ]; then
  CFFFILE="Sherpa3_${ORIGIN_NAME}_cff.py"
fi

CHECKSUM=$(md5sum "${GRIDPACK}" | awk '{print $1}')

# +-----------------------------------------------------------------------------------------------+
# get the runcard (from inside the gridpack, unless given explicitly)
# +-----------------------------------------------------------------------------------------------+

TMPDIR_CFF=$(mktemp -d)
trap 'rm -rf "${TMPDIR_CFF}"' EXIT

if [ -z "${SH_YML}" ]; then
  tar -xOJf "${GRIDPACK}" ./Sherpa.yaml > "${TMPDIR_CFF}/Sherpa.yaml" 2>/dev/null \
    || { echo "mkSherpa3cff: could not extract ./Sherpa.yaml from ${GRIDPACK}"; exit 1; }
  SH_YML="${TMPDIR_CFF}/Sherpa.yaml"
fi
if [ ! -f "${SH_YML}" ]; then
  echo "mkSherpa3cff: runcard '${SH_YML}' not found"
  exit 1
fi

if [ -n "${WEIGHTS_FILE}" ] && [ ! -s "${WEIGHTS_FILE}" ]; then
  echo "mkSherpa3cff: weight name file '${WEIGHTS_FILE}' not found or empty"
  exit 1
fi

echo "mkSherpa3cff: gridpack  = ${GRIDPACK}"
echo "mkSherpa3cff: process   = ${ORIGIN_NAME}"
echo "mkSherpa3cff: md5       = ${CHECKSUM}"
echo "mkSherpa3cff: runcard   = ${SH_YML}"
echo "mkSherpa3cff: cff       = ${CFFFILE}"
if [ -n "${WEIGHTS_FILE}" ]; then
  echo "mkSherpa3cff: weights   = ${WEIGHTS_FILE} ($(wc -l < "${WEIGHTS_FILE}") names)"
else
  echo "mkSherpa3cff: weights   = none given (-w/--weights); weights will be unnamed in GEN output"
fi

# +-----------------------------------------------------------------------------------------------+
# build the cff
# +-----------------------------------------------------------------------------------------------+

# embed the YAML lines verbatim (indentation is significant in YAML!),
# only escaping '\' and '"' and quoting each line (no trailing comma on the last one)
sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' "${SH_YML}" \
  | sed -e 's/^/                                "/;s/$/\",/' \
  | sed -e '$s/\",$/"/' > "${TMPDIR_CFF}/card.tmp"

if [ -e "${CFFFILE}" ]; then rm "${CFFFILE}"; fi
touch "${CFFFILE}"

echo "import FWCore.ParameterSet.Config as cms"                          >> ${CFFFILE}
echo "import os"                                                         >> ${CFFFILE}
echo ""                                                                  >> ${CFFFILE}
echo "source = cms.Source(\"EmptySource\")"                              >> ${CFFFILE}
echo ""                                                                  >> ${CFFFILE}
echo "generator = cms.EDFilter(\"Sherpa3GeneratorFilter\","              >> ${CFFFILE}
echo "  maxEventsToPrint = cms.int32(0),"                                >> ${CFFFILE}
echo "  filterEfficiency = cms.untracked.double(1.0),"                   >> ${CFFFILE}
echo "  crossSection = cms.untracked.double(-1),"                        >> ${CFFFILE}
echo "  Sherpa3Process = cms.string('"${ORIGIN_NAME}"'),"                >> ${CFFFILE}
echo "  GridpackLocation = cms.string('"${GRIDPACK}"'),"                 >> ${CFFFILE}
echo "  GridpackChecksum = cms.string('"${CHECKSUM}"'),"                 >> ${CFFFILE}
if [ -n "${WEIGHTS_FILE}" ]; then
  # nominal + bookkeeping entries first (the hadronizer normalises index 0),
  # everything else is a variation and gets divided by the weight normalisation.
  # Next to each list a SherpaRenamed* list documents the EDM-facing names the
  # hadronizer stores ('+' -> 'p', '-' -> 'm'; downstream CMSSW consumers such
  # as RivetAnalyzer mangle characters outside [A-Za-z0-9._=], which would make
  # Sherpa's polarisation weights collide). Same order as the original list.
  {
    echo "  SherpaWeightsBlock = cms.PSet("
    echo "    SherpaWeights = cms.vstring("
    (grep -x 'Weight' "${WEIGHTS_FILE}"; grep -E '^(EXTRA__|IRREG__)' "${WEIGHTS_FILE}") \
      | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    echo "    ),"
    echo "    # EDM-facing names, one per SherpaWeights entry in the same order"
    echo "    SherpaRenamedWeights = cms.vstring("
    (grep -x 'Weight' "${WEIGHTS_FILE}"; grep -E '^(EXTRA__|IRREG__)' "${WEIGHTS_FILE}") \
      | sed -e 's/+/p/g; s/-/m/g' | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    echo "    ),"
    echo "    SherpaVariationWeights = cms.vstring("
    grep -vx 'Weight' "${WEIGHTS_FILE}" | grep -vE '^(EXTRA__|IRREG__)' \
      | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    echo "    ),"
    echo "    # EDM-facing names, one per SherpaVariationWeights entry in the same order"
    echo "    SherpaRenamedVariationWeights = cms.vstring("
    grep -vx 'Weight' "${WEIGHTS_FILE}" | grep -vE '^(EXTRA__|IRREG__)' \
      | sed -e 's/+/p/g; s/-/m/g' | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    echo "    )"
    echo "  ),"
  }                                                                      >> ${CFFFILE}
fi
echo "  Sherpa3Parameters = cms.PSet(parameterSets = cms.vstring("       >> ${CFFFILE}
echo "                             \"Run\"),"                            >> ${CFFFILE}
echo "                              Run = cms.vstring("                  >> ${CFFFILE}
cat "${TMPDIR_CFF}/card.tmp"                                             >> ${CFFFILE}
echo "                              )"                                   >> ${CFFFILE}
echo "                             )"                                    >> ${CFFFILE}
echo ")"                                                                 >> ${CFFFILE}
echo ""                                                                  >> ${CFFFILE}
echo "ProductionFilterSequence = cms.Sequence(generator)"                >> ${CFFFILE}
echo ""                                                                  >> ${CFFFILE}

echo "mkSherpa3cff: cff created: ${CFFFILE}"
