#!/bin/bash
#
#  file:        mkSherpa3Gridpack.sh
#  description: BASH script to produce a Sherpa3 gridpack (process libraries +
#               cross sections) from a Sherpa 3 runcard in YAML format.
#               Must be run inside a CMSSW environment (cmsenv) with the
#               sherpa3 tool set up, so that SHERPA3_* / SCRAM_ARCH /
#               CMSSW_VERSION are defined.
#
#  usage:       mkSherpa3Gridpack.sh -yml <runcard.yaml> [options]
#
#  options: -exe, --sh_exe   path   Sherpa3 executable
#                                   (default: the CMSSW sherpa3 from the sherpa3 tool)
#           -yml, --sh_yml   path   runcard in YAML format; it is copied to
#                                   "Sherpa.yaml" inside the running directory
#           -o,   --outpath  path   running directory; all output goes there
#                                   (default: TEMP_SHERPA; deleted & recreated
#                                   if it exists)
#           -n,   --nthreads N      number of MPI ranks for the cross section
#                                   integration run (default: 1 -> no mpiexec)
#           -ol,  --openloops path  OpenLoops prefix used for OL_PREFIX in
#                                   Sherpa.yaml (default: the openloops CMSSW
#                                   points to, CMS_OPENLOOPS_PREFIX; only set
#                                   when the runcard mentions OpenLoops)
#           -so,  --sh_opt "opts"   additional Sherpa options (string), passed
#                                   to Sherpa for the cross section
#                                   integration run
#           -ht,  --hyperthread     allow using hyperthreads: NTHREADS may go
#                                   beyond the physical cores up to the total
#                                   number of threads (mpiexec is run with
#                                   --use-hwthread-cpus)
#           -nw,  --no-weights      skip the weight-name probe (see notes); the
#                                   cff is then written without a
#                                   SherpaWeightsBlock and the event weights end
#                                   up in the GEN output unnamed
#           -v,   --verbose         also print the Sherpa/makelibs running
#                                   output to stdout (default: log file only)
#           -h,   --help            display this help and exit
#
#  notes:       AMEGIC usage is detected automatically: if the runcard mentions
#               Amegic (e.g. in ME_GENERATORS), or if the library-writing step
#               produced ./makelibs, the process libraries are compiled with
#               ./makelibs. If the runcard contains "AMEGIC_LIBRARY_MODE: 0",
#               ./makelibs is run with "-m" (one library per process).
#
#               Weight names: after the gridpack is packed, one event is
#               generated with Sherpa's HepMC3 output enabled and the weight
#               names are read back from it. They are written into the cff as a
#               SherpaWeightsBlock, so the GEN job can label its weights. This
#               is the only way to get the names right: they depend on the
#               runcard (scale/PDF variations, associated contributions) and
#               Sherpa only materialises them when it fills an event. Nothing
#               downstream carries them otherwise - HepMC3Product stores no
#               GenRunInfo and GenEventInfoProduct3 stores values only, so
#               GenLumiInfoHeader (fed from this block) is the only channel.
#               The block also documents SherpaRenamedWeights /
#               SherpaRenamedVariationWeights: the EDM-facing names the
#               hadronizer stores ('+' -> 'p', '-' -> 'm'), one per entry of
#               the matching Sherpa list, in the same order.
#
#               OL_PREFIX is baked into the cff for the CMSSW release used to
#               build the gridpack. At event-generation time the hadronizer
#               overrides it with the OpenLoops of the running release
#               (CMS_OPENLOOPS_PREFIX), unless the fragment opts out with
#               FIXED_OL_PREFIX = cms.bool(True) in Sherpa3Parameters.
#
#  output:      Sherpa3_<runcard_name>_<SCRAM_ARCH>_<CMSSW_VERSION>_tarball.tar.xz
#               in the directory where the script is invoked, plus the CMSSW
#               python fragment Sherpa3_<runcard_name>_cff.py for it and the
#               plain list of weight names Sherpa3_<runcard_name>_weights.txt.
#               A log file with all stdout/stderr is stored inside the gridpack.

set -uo pipefail

# +-----------------------------------------------------------------------------------------------+
# defaults & argument parsing
# +-----------------------------------------------------------------------------------------------+

SH_EXE=""
SH_YML=""
FLAG_AMEGIC=0
OUTPATH="SHERPA3TEMP"
NTHREADS=1
OL_PREFIX=""
OL_EXPLICIT=0
SH_OPT=""
USE_HT=0
EVTGEN_DIR="./SHERPA3GEN" # Sherpa3 event generation directory in cff
GET_WEIGHTS=1
VERBOSE=0

# remember the original command line (for the log)
ORIG_CMD="$0 $*"

# print the header comment block (everything after the shebang up to the first
# non-comment line), so the help never goes stale when the header grows
print_help() { awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -exe|--sh_exe)     SH_EXE="$2";      shift 2 ;;
    -yml|--sh_yml)     SH_YML="$2";      shift 2 ;;
    -o|--outpath)      OUTPATH="$2";     shift 2 ;;
    -n|--nthreads)     NTHREADS="$2";    shift 2 ;;
    -ol|--openloops)   OL_PREFIX="$2"; OL_EXPLICIT=1; shift 2 ;;
    -so|--sh_opt)      SH_OPT="$2";      shift 2 ;;
    -ht|--hyperthread) USE_HT=1;         shift   ;;
    -nw|--no-weights)  GET_WEIGHTS=0;    shift   ;;
    -eg|--evtgendir)   EVTGEN_DIR="$2";  shift 2 ;;
    -v|--verbose)      VERBOSE=1;        shift   ;;
    -h|--help)         print_help;       exit 0  ;;
    *) echo "mkSherpa3Gridpack: unknown option '$1' (use -h for help)"; exit 1 ;;
  esac
done

# +-----------------------------------------------------------------------------------------------+
# sanity checks
# +-----------------------------------------------------------------------------------------------+

if [ -z "${SH_YML}" ]; then
  echo "mkSherpa3Gridpack: no runcard given, use -yml/--sh_yml (use -h for help)"
  exit 1
fi
if [ ! -f "${SH_YML}" ]; then
  echo "mkSherpa3Gridpack: runcard '${SH_YML}' not found"
  exit 1
fi
# make the runcard path absolute (the script changes into OUTPATH later)
SH_YML="$(cd "$(dirname "${SH_YML}")" && pwd)/$(basename "${SH_YML}")"

if [ -z "${SHERPA3_SHARE_PATH:-}" ]; then
  echo "mkSherpa3Gridpack: SHERPA3_SHARE_PATH is not set."
  echo "                   Run this script in a CMSSW environment (cmsenv) with the sherpa3 tool set up."
  exit 1
fi

# default Sherpa3 executable: the CMSSW sherpa3 (derived from the sherpa3 tool paths)
if [ -z "${SH_EXE}" ]; then
  SH_EXE="$(cd "${SHERPA3_SHARE_PATH}/../.." && pwd)/bin/Sherpa3"
fi
if [ ! -x "${SH_EXE}" ]; then
  echo "mkSherpa3Gridpack: Sherpa3 executable '${SH_EXE}' not found or not executable"
  exit 1
fi

if [ -z "${SCRAM_ARCH:-}" ] || [ -z "${CMSSW_VERSION:-}" ]; then
  echo "mkSherpa3Gridpack: SCRAM_ARCH and/or CMSSW_VERSION not set; run inside cmsenv."
  exit 1
fi

# OpenLoops prefix for OL_PREFIX in Sherpa.yaml.
# Only needed when the runcard uses OpenLoops; -ol forces it.
if grep -qi "openloops" "${SH_YML}"; then
  FLAG_OPENLOOPS=1
else
  FLAG_OPENLOOPS=0
fi
if [ "${OL_EXPLICIT}" -eq 1 ] && [ "${FLAG_OPENLOOPS}" -eq 0 ]; then
  echo "mkSherpa3Gridpack: WARNING: -ol given but runcard '${SH_YML}' does not mention OpenLoops"
fi
if [ "${FLAG_OPENLOOPS}" -eq 0 ] && [ "${OL_EXPLICIT}" -eq 0 ]; then
  OL_PREFIX=""
elif [ -z "${OL_PREFIX}" ]; then
  OL_PREFIX="${CMS_OPENLOOPS_PREFIX:-}"
fi
if [ -n "${OL_PREFIX}" ] && [ ! -d "${OL_PREFIX}" ]; then
  echo "mkSherpa3Gridpack: OpenLoops prefix '${OL_PREFIX}' is not a directory"
  exit 1
fi

# +-----------------------------------------------------------------------------------------------+
# automatic AMEGIC detection (from the runcard)
# +-----------------------------------------------------------------------------------------------+

if grep -qi "amegic" "${SH_YML}"; then
  FLAG_AMEGIC=1
fi
MKLIBS_OPTS=""
if grep -Eq "^[[:space:]]*AMEGIC_LIBRARY_MODE:[[:space:]]*0([[:space:]]|$)" "${SH_YML}"; then
  MKLIBS_OPTS="-m"
fi

ORIGIN_NAME=$(basename "${SH_YML}")
ORIGIN_NAME="${ORIGIN_NAME%.*}"
GRIDPACK="$(pwd)/Sherpa3_${ORIGIN_NAME}_${SCRAM_ARCH}_${CMSSW_VERSION}_tarball.tar.xz"
CFFFILE="$(pwd)/Sherpa3_${ORIGIN_NAME}_cff.py"
WEIGHTS_FILE="$(pwd)/Sherpa3_${ORIGIN_NAME}_weights.txt"
LOG="$(pwd)/${ORIGIN_NAME}.log"
rm -f "${WEIGHTS_FILE}"
: > "${LOG}"

# +-----------------------------------------------------------------------------------------------+
# logging helpers
# +-----------------------------------------------------------------------------------------------+

logecho() {
  # print to stdout and append to the log file
  echo "$@" | tee -a "${LOG}"
}

run_step() {
  # log a command and run it; append all stdout/stderr to the log file
  # (and print it live when running verbose)
  echo "" >> "${LOG}"
  logecho "+ $*"
  if [ "${VERBOSE}" = "1" ]; then
    "$@" 2>&1 | tee -a "${LOG}"
    return "${PIPESTATUS[0]}"
  else
    "$@" >> "${LOG}" 2>&1
    return $?
  fi
}

get_weight_names() {
    # Generate a single event with Sherpa's HepMC3 output enabled and read the
    # weight names back from the "W" record of the HepMC3 ASCII file. Must be
    # called from inside OUTPATH, after the integration run. Writes the names,
    # one per line, to ${WEIGHTS_FILE}; leaves it absent on failure.
    logecho "##############################"
    logecho ">>> PROBING WEIGHT NAMES"
    logecho "##############################"

    local probe_card="weights_probe.yaml"
    local probe_out="weights_probe"

    cp Sherpa.yaml "${probe_card}" || { logecho " could not create ${probe_card}"; return 1; }
    echo "EVENT_OUTPUT: HepMC3[${probe_out}]" >> "${probe_card}"

    # one event only; Sherpa loads libSherpaHepMC3Output on demand
    if ! run_step "${SH_EXE}" "EVENT_OUTPUT: HepMC3[${probe_out}]" ${SH_OPT} -e 1 "${probe_card}"; then
      logecho " WARNING: the weight-name probe run failed, see the log."
      logecho "          The cff will be written without a SherpaWeightsBlock."
      return 1
    fi

    # Sherpa writes <name>.gz (or an uncompressed variant, depending on build)
    local probe_file
    probe_file=$(ls -1 "${probe_out}".gz "${probe_out}".hepmc3 "${probe_out}" 2>/dev/null | head -1)
    if [ -z "${probe_file}" ]; then
      logecho " WARNING: no HepMC3 output found from the weight-name probe."
      logecho "          The cff will be written without a SherpaWeightsBlock."
      return 1
    fi

    # the W record lists the names separated by '\|'
    local reader="cat"
    case "${probe_file}" in *.gz) reader="zcat" ;; esac
    ${reader} "${probe_file}" 2>/dev/null | grep -m1 '^W ' | sed -e 's/^W //' \
      | tr '\\' '\n' | sed -e 's/^|//' | grep -v '^$' > "${WEIGHTS_FILE}"

    if [ ! -s "${WEIGHTS_FILE}" ]; then
      logecho " WARNING: could not read any weight name from ${probe_file}."
      logecho "          (HEPMC_USE_NAMED_WEIGHTS disabled in the runcard?)"
      logecho "          The cff will be written without a SherpaWeightsBlock."
      rm -f "${WEIGHTS_FILE}"
      return 1
    fi

    logecho " found $(wc -l < "${WEIGHTS_FILE}") weight names -> ${WEIGHTS_FILE} -> weights_list.txt"
    rm -f "${probe_card}" "${probe_out}".gz "${probe_out}".hepmc3 "${probe_out}"
    return 0
}

write_weights_block() {
    # Turn ${WEIGHTS_FILE} into a SherpaWeightsBlock PSet on stdout.
    # The nominal weight and the bookkeeping entries (EXTRA__*, IRREG__*) go
    # into SherpaWeights, everything else is a variation and goes into
    # SherpaVariationWeights, which the hadronizer divides by the weight
    # normalisation for unweighted events. "Weight" must stay first: the
    # hadronizer normalises index 0 of the rearranged list.
    # Next to each list a SherpaRenamed* list documents the EDM-facing names
    # the hadronizer stores ('+' -> 'p', '-' -> 'm'; downstream CMSSW consumers
    # such as RivetAnalyzer mangle characters outside [A-Za-z0-9._=], which
    # would make Sherpa's polarisation weights collide). Same order as the
    # original list.
    local base var
    base=$( (grep -x 'Weight' "${WEIGHTS_FILE}"; grep -E '^(EXTRA__|IRREG__)' "${WEIGHTS_FILE}") )
    var=$(grep -vx 'Weight' "${WEIGHTS_FILE}" | grep -vE '^(EXTRA__|IRREG__)')

    echo "  SherpaWeightsBlock = cms.PSet("
    echo "    SherpaWeights = cms.vstring("
    echo "${base}" | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    echo "    ),"
    echo "    # EDM-facing names, one per SherpaWeights entry in the same order"
    echo "    SherpaRenamedWeights = cms.vstring("
    echo "${base}" | sed -e 's/+/p/g; s/-/m/g' | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    echo "    ),"
    echo "    SherpaVariationWeights = cms.vstring("
    if [ -n "${var}" ]; then
      echo "${var}" | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    fi
    echo "    ),"
    echo "    # EDM-facing names, one per SherpaVariationWeights entry in the same order"
    echo "    SherpaRenamedVariationWeights = cms.vstring("
    if [ -n "${var}" ]; then
      echo "${var}" | sed -e 's/+/p/g; s/-/m/g' | sed -e "s/^/      '/;s/$/',/" -e '$s/,$//'
    fi
    echo "    )"
    echo "  ),"
}

create_cff() {
    logecho "##############################"
    logecho ">>> CREATING CFF"
    logecho "##############################"

    TMPDIR_CFF=$(mktemp -d)
    trap 'rm -rf "${TMPDIR_CFF}"' EXIT

    tar -xOJf "${GRIDPACK}" ./Sherpa.yaml > "${TMPDIR_CFF}/Sherpa.yaml" 2>/dev/null \
      || { logecho " could not extract ./Sherpa.yaml from ${GRIDPACK}"; exit 1; }

    CHECKSUM=$(md5sum "${GRIDPACK}" | awk '{print $1}')

    # embed the YAML lines verbatim (indentation is significant in YAML!),
    # only escaping '\' and '"' and quoting each line (no trailing comma on the last one)
    sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' "${TMPDIR_CFF}/Sherpa.yaml" \
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
    echo "  EvtGenDirectory = cms.string('"${EVTGEN_DIR}"'),"                >> ${CFFFILE}
    if [ -s "${WEIGHTS_FILE}" ]; then
      write_weights_block                                                    >> ${CFFFILE}
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
}

# +-----------------------------------------------------------------------------------------------+
# workflow
# +-----------------------------------------------------------------------------------------------+

# cap the number of MPI ranks / compile jobs: by default at the number of
# physical CPU cores (OpenMPI counts physical cores as slots); with
# -ht/--hyperthread at the total number of threads (hyperthreads included)
PHYS_CORES=$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)
if [ -z "${PHYS_CORES}" ] || [ "${PHYS_CORES}" -lt 1 ] 2>/dev/null; then
  PHYS_CORES=$(nproc 2>/dev/null || echo 1)
fi
TOTAL_THREADS=$(nproc 2>/dev/null || echo "${PHYS_CORES}")
MPI_HT_OPT=""
if [ "${USE_HT}" = "1" ]; then
  MAX_RANKS=${TOTAL_THREADS}
  MPI_HT_OPT="--use-hwthread-cpus"
else
  MAX_RANKS=${PHYS_CORES}
fi
if [ "${NTHREADS}" -gt "${MAX_RANKS}" ]; then
  logecho " requested nthreads=${NTHREADS} exceeds the maximum ranks (${MAX_RANKS}); using ${MAX_RANKS}"
  NTHREADS=${MAX_RANKS}
fi

logecho "##############################"
logecho ">>> START INFO"
logecho "##############################"
logecho " command        = ${ORIG_CMD}"
logecho " date           = $(date)"
logecho " runcard        = ${SH_YML}"
logecho " Sherpa3 exe    = ${SH_EXE}"
logecho " FLAG_AMEGIC    = ${FLAG_AMEGIC} (auto-detected)"
logecho " makelibs opts  = '${MKLIBS_OPTS}'"
logecho " OL_PREFIX      = ${OL_PREFIX}"
logecho " Sherpa options = '${SH_OPT}'"
logecho " outpath        = ${OUTPATH}"
logecho " nthreads       = ${NTHREADS} (max ${MAX_RANKS}, hyperthreads=${USE_HT})"
logecho " gridpack       = ${GRIDPACK}"
logecho " evtgendir      = ${EVTGEN_DIR}"

# (1) create outpath (delete & recreate if it exists)
if [ -e "${OUTPATH}" ]; then
  logecho " removing existing '${OUTPATH}'"
  rm -rf "${OUTPATH}" || exit 1
fi
mkdir -p "${OUTPATH}" || exit 1

logecho "##############################"
logecho ">>> RUNNING"
logecho "##############################"

cd "${OUTPATH}" || exit 1

# (2) copy runcard to Sherpa.yaml
cp "${SH_YML}" Sherpa.yaml || { logecho " failed to copy runcard"; exit 1; }

# (2b) set up OL_PREFIX in Sherpa.yaml (replace an existing setting, else append)
if [ -n "${OL_PREFIX}" ]; then
  if grep -Eq "^[[:space:]]*OL_PREFIX:" Sherpa.yaml; then
    sed -i "s|^[[:space:]]*OL_PREFIX:.*|OL_PREFIX: ${OL_PREFIX}|" Sherpa.yaml \
      || { logecho " failed to set OL_PREFIX in Sherpa.yaml"; exit 1; }
  else
    echo "OL_PREFIX: ${OL_PREFIX}" >> Sherpa.yaml \
      || { logecho " failed to set OL_PREFIX in Sherpa.yaml"; exit 1; }
  fi
  logecho " OL_PREFIX set to ${OL_PREFIX} in Sherpa.yaml"
fi

# (3) initiate: write process source code
run_step "${SH_EXE}" ${SH_OPT} -I Sherpa.yaml || { logecho " library-writing step failed, see ${LOG}"; exit 1; }

# if the library-writing step produced ./makelibs, process libraries must be
# compiled regardless of the runcard-based detection
if [ -x ./makelibs ] && [ "${FLAG_AMEGIC}" != "1" ]; then
  logecho " ./makelibs was produced, enabling AMEGIC library compilation"
  FLAG_AMEGIC=1
fi

# (4) compile process libraries (only if AMEGIC is used)
if [ "${FLAG_AMEGIC}" = "1" ]; then
  if [ -x ./makelibs ]; then
    run_step ./makelibs ${MKLIBS_OPTS} -j ${NTHREADS} || { logecho " makelibs failed, see ${LOG}"; exit 1; }
  else
    logecho " WARNING: AMEGIC detected in runcard but no ./makelibs was produced;"
    logecho "          no process libraries to compile, continuing."
  fi
fi

# (5) cross section integration run
# (MPI_HT_OPT and additional Sherpa options from -so/--sh_opt are word-split on purpose)
if [ "${NTHREADS}" -gt 1 ]; then
  run_step mpiexec ${MPI_HT_OPT} -np "${NTHREADS}" "${SH_EXE}" ${SH_OPT} -e 0 Sherpa.yaml \
    || { logecho " integration run failed, see ${LOG}"; exit 1; }
else
  run_step "${SH_EXE}" ${SH_OPT} -e 0 Sherpa.yaml \
    || { logecho " integration run failed, see ${LOG}"; exit 1; }
fi

# (6) probe the weight names (after packing, so the probe leaves no trace in
# the gridpack; a failure here is not fatal, the cff is just written without
# a SherpaWeightsBlock)
if [ "${GET_WEIGHTS}" = "1" ]; then
  get_weight_names || true
else
  logecho " weight-name probe skipped (-nw/--no-weights)"
fi
cp "${WEIGHTS_FILE}" "weights_list.txt"

# (7) compress to gridpack (log file included)
run_step echo "mkSherpa3Gridpack finished at $(date)"
ORIGIN_LOG="${LOG}"
mv "${ORIGIN_LOG}" gridpack_generation.log
# from here on, append to the moved log inside OUTPATH
LOG="gridpack_generation.log"
logecho "##############################"
logecho ">>> PACKING"
logecho "##############################"
logecho " packing '${GRIDPACK}' ..."
cp "${LOG}" "${ORIGIN_LOG}"
tar -cJf "${GRIDPACK}" . || { logecho " failed to create gridpack"; exit 1; }

cd ..
# back in the invocation directory: append further log lines to the copied log
LOG="${ORIGIN_LOG}"
logecho "[ mkSherpa3Gridpack ] gridpack created: ${GRIDPACK}"

# (8) generate the CMSSW python fragment (cff) for the gridpack
create_cff
logecho "[ mkSherpa3Gridpack ] cff created: ${CFFFILE}"
if [ -s "${WEIGHTS_FILE}" ]; then
  logecho "[ mkSherpa3Gridpack ] weight names: ${WEIGHTS_FILE} ($(wc -l < "${WEIGHTS_FILE}") entries, embedded in the cff)"
else
  logecho "[ mkSherpa3Gridpack ] WARNING: no weight names available; the GEN output will store the"
  logecho "                      event weights unnamed. Re-run without -nw/--no-weights to fix."
fi
