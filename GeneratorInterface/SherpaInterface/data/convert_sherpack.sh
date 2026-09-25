#!/bin/bash
# convert_sherpack.sh
# Converts Sherpa sherpacks between:
#   new format: sherpa_<PROCESS>_<SCRAM_ARCH>_<CMSSW_VERSION>.tar.xz
#   old format: sherpa_<PROCESS>.tgz
# and generates the corresponding _cff.py from Run.dat / MPI_Cross_Sections
# inside the sherpack.
#
# Usage:
#   new_to_old:  convert_sherpack.sh new_to_old sherpa_DY_MASTER_el8_amd64_gcc12_CMSSW_14_0_21.tar.xz
#   old_to_new:  convert_sherpack.sh old_to_new sherpa_DY_MASTER.tgz
#                (requires SCRAM_ARCH and CMSSW_VERSION set in the environment)

set -euo pipefail

usage() {
    echo "Usage: $0 <mode> <sherpack_file>"
    echo ""
    echo "  Modes:"
    echo "    new_to_old   sherpa_<PROCESS>_<SCRAM_ARCH>_<CMSSW_VERSION>.tar.xz"
    echo "    old_to_new   sherpa_<PROCESS>.tgz"
    echo ""
    echo "For old_to_new, SCRAM_ARCH and CMSSW_VERSION must be set in the environment"
    echo "(they are set automatically inside a cmsenv shell)."
    exit 1
}

die() { echo "Error: $*" >&2; exit 1; }

[ $# -eq 2 ] || usage

MODE="$1"
INPUT="$2"

[[ "$MODE" == "new_to_old" || "$MODE" == "old_to_new" ]] \
    || die "Unknown mode '$MODE'. Use 'new_to_old' or 'old_to_new'."
[ -f "$INPUT" ] || die "File '$INPUT' not found"

TMPDIR_WORK=$(mktemp -d)
trap 'rm -rf "$TMPDIR_WORK"' EXIT

# ── generate_cff ──────────────────────────────────────────────────────────────
# Args: sherpa_process  sherpack_location  checksum  tmpdir  output_cff
generate_cff() {
    local sherpa_process="$1"
    local sherpack_location="$2"
    local checksum="$3"
    local tmpdir="$4"
    local output_cff="$5"

    echo "Generating ${output_cff} ..."

    local run_dat
    run_dat=$(find "$tmpdir" -name "Run.dat" | head -1 || true)
    [ -n "$run_dat" ] || echo "  Warning: Run.dat not found in sherpack — Run vstring will be empty"

    # Pass paths via environment; use single-quoted PYEOF so Python code is
    # not subject to bash expansion (Run.dat may contain { } * etc.)
    SHERPA_PROCESS="$sherpa_process" \
    SHERPACK_LOCATION="$sherpack_location" \
    CHECKSUM="$checksum" \
    RUN_DAT="${run_dat}" \
    OUTPUT_CFF="$output_cff" \
    python3 << 'PYEOF'
import os

sherpa_process    = os.environ['SHERPA_PROCESS']
sherpack_location = os.environ['SHERPACK_LOCATION']
checksum          = os.environ['CHECKSUM']
run_dat_path      = os.environ.get('RUN_DAT', '')
output_cff        = os.environ['OUTPUT_CFF']

def read_all_lines(path):
    if not path:
        return []
    try:
        with open(path) as f:
            return [l.rstrip('\n') for l in f]
    except IOError:
        return []

run_lines = read_all_lines(run_dat_path)

INDENT = ' ' * 32

# MPI_Cross_Sections is always these fixed 3 lines
mpi_vstring = (
    '%s" MPIs in Sherpa, Model = Amisic:",\n'
    '%s" semihard xsec = 39.2965 mb,",\n'
    '%s" non-diffractive xsec = 17.0318 mb with nd factor = 0.3142."'
) % (INDENT, INDENT, INDENT)

def make_vstring(lines, placeholder):
    if not lines:
        return '%s" %s"' % (INDENT, placeholder)
    parts = []
    for i, line in enumerate(lines):
        comma = ',' if i < len(lines) - 1 else ''
        parts.append('%s" %s"%s' % (INDENT, line, comma))
    return '\n'.join(parts)

run_vstring = make_vstring(run_lines, 'Run.dat not found in sherpack')

# Use %-formatting so Sherpa's { } braces in run_vstring are not misread.
template = """\
import FWCore.ParameterSet.Config as cms
import os

source = cms.Source("EmptySource")

generator = cms.EDFilter("SherpaGeneratorFilter",
  maxEventsToPrint = cms.int32(0),
  filterEfficiency = cms.untracked.double(1.0),
  crossSection = cms.untracked.double(-1),
  SherpaProcess = cms.string('%s'),
  SherpackLocation = cms.string('%s'),
  SherpackChecksum = cms.string('%s'),
  FetchSherpack = cms.bool(True),
  SherpaPath = cms.string('./'),
  SherpaPathPiece = cms.string('./'),
  SherpaResultDir = cms.string('Result'),
  SherpaDefaultWeight = cms.double(1.0),
  SherpaParameters = cms.PSet(parameterSets = cms.vstring(
                             "MPI_Cross_Sections",
                             "Run"),
                              MPI_Cross_Sections = cms.vstring(
%s
                                                  ),
                              Run = cms.vstring(
%s
                                                  ),
                             )
)

ProductionFilterSequence = cms.Sequence(generator)
"""

content = template % (sherpa_process, sherpack_location, checksum,
                      mpi_vstring, run_vstring)

with open(output_cff, 'w') as f:
    f.write(content)
PYEOF

    echo "  Generated: ${output_cff}"
}

# ── new_to_old ────────────────────────────────────────────────────────────────
if [[ "$MODE" == "new_to_old" ]]; then

    [[ "$INPUT" == *.tar.xz ]] \
        || die "new_to_old expects a .tar.xz file, got: $INPUT"

    BASENAME=$(basename "${INPUT%.tar.xz}")

    [[ "$BASENAME" == *_CMSSW_* ]] || die "Cannot find _CMSSW_ in filename '$BASENAME'"
    CMSSW_VERSION="CMSSW_${BASENAME##*_CMSSW_}"
    STEM="${BASENAME%_${CMSSW_VERSION}}"

    # SCRAM_ARCH: last 3 underscore-separated tokens of the stem
    PART3="${STEM##*_}";    REST="${STEM%_$PART3}"
    PART2="${REST##*_}";    REST="${REST%_$PART2}"
    PART1="${REST##*_}";    PROCESS="${REST%_$PART1}"
    SCRAM_ARCH="${PART1}_${PART2}_${PART3}"

    # SherpaProcess: strip "sherpa_" prefix and optional "_MASTER" suffix
    SHERPA_PROC="${PROCESS#sherpa_}"
    SHERPA_PROC="${SHERPA_PROC%_MASTER}"

    OUTPUT="${PROCESS}.tgz"
    MD5FILE="${PROCESS}.md5"
    CFF_PY="${PROCESS}_cff.py"

    # SherpackLocation for old format is the parent directory of the tgz
    SHERPACK_DIR="$(pwd)/"

    echo "=== new_to_old ==="
    echo "  Input:            $INPUT"
    echo "  Output:           $OUTPUT"
    echo "  CFF:              $CFF_PY"
    echo "  MD5 file:         $MD5FILE"
    echo "  SherpaProcess:    $SHERPA_PROC"
    echo "  SCRAM_ARCH:       $SCRAM_ARCH"
    echo "  CMSSW_VERSION:    $CMSSW_VERSION"
    echo "  SherpackLocation: $SHERPACK_DIR"
    echo ""

    echo "Extracting $INPUT ..."
    tar -xJf "$INPUT" -C "$TMPDIR_WORK"

    echo "Repacking as $OUTPUT (gzip) ..."
    tar -czf "$OUTPUT" -C "$TMPDIR_WORK" .

    echo "Computing MD5 checksum ..."
    MD5_LINE=$(md5sum "$OUTPUT")           # e.g. "c0f81ea6...  sherpa_DY_MASTER.tgz"
    MD5=$(echo "$MD5_LINE" | awk '{print $1}')
    echo "$MD5_LINE" > "$MD5FILE"
    echo "  $MD5_LINE"
    echo "  Saved to: $MD5FILE"
    echo ""

    generate_cff "$SHERPA_PROC" "$SHERPACK_DIR" "$MD5" "$TMPDIR_WORK" "$CFF_PY"

    echo ""
    echo "=========================================================="
    echo "WARNING: update SherpackLocation and SherpackChecksum"
    echo "=========================================================="
    echo "In ${CFF_PY} (or the _GEN.py that imports it):"
    echo ""
    echo "  SherpackLocation = cms.string('${SHERPACK_DIR}'),"
    echo "  SherpackChecksum = cms.string('${MD5}'),"
    echo ""
    echo "If you move ${OUTPUT} to a different directory, update"
    echo "SherpackLocation to that directory's path."
    echo "Previously pointed to: ${INPUT}"
    echo "=========================================================="

# ── old_to_new ────────────────────────────────────────────────────────────────
elif [[ "$MODE" == "old_to_new" ]]; then

    [[ "$INPUT" == *.tgz ]] \
        || die "old_to_new expects a .tgz file, got: $INPUT"

    SCRAM_ARCH="${SCRAM_ARCH:-}"
    CMSSW_VERSION="${CMSSW_VERSION:-}"
    [ -n "$SCRAM_ARCH" ]    || die "SCRAM_ARCH is not set. Run 'cmsenv' first, or: export SCRAM_ARCH=el8_amd64_gcc12"
    [ -n "$CMSSW_VERSION" ] || die "CMSSW_VERSION is not set. Run 'cmsenv' first, or: export CMSSW_VERSION=CMSSW_14_0_21"

    PROCESS=$(basename "${INPUT%.tgz}")
    OUTPUT="${PROCESS}_${SCRAM_ARCH}_${CMSSW_VERSION}.tar.xz"
    OUTPUT_FULLPATH="$(pwd)/${OUTPUT}"
    CFF_PY="${PROCESS}_cff.py"

    SHERPA_PROC="${PROCESS#sherpa_}"
    SHERPA_PROC="${SHERPA_PROC%_MASTER}"

    echo "=== old_to_new ==="
    echo "  Input:            $INPUT"
    echo "  Output:           $OUTPUT_FULLPATH"
    echo "  CFF:              $CFF_PY"
    echo "  SherpaProcess:    $SHERPA_PROC"
    echo "  SCRAM_ARCH:       $SCRAM_ARCH"
    echo "  CMSSW_VERSION:    $CMSSW_VERSION"
    echo ""

    echo "Extracting $INPUT ..."
    tar -xzf "$INPUT" -C "$TMPDIR_WORK"

    echo "Repacking as $OUTPUT (xz) ..."
    tar -cJf "$OUTPUT" -C "$TMPDIR_WORK" .

    echo "Computing MD5 checksum ..."
    MD5_LINE=$(md5sum "$OUTPUT")
    MD5=$(echo "$MD5_LINE" | awk '{print $1}')
    echo "  $MD5_LINE"
    echo ""

    generate_cff "$SHERPA_PROC" "$OUTPUT_FULLPATH" "$MD5" "$TMPDIR_WORK" "$CFF_PY"

    echo ""
    echo "=========================================================="
    echo "WARNING: update SherpackLocation and SherpackChecksum"
    echo "=========================================================="
    echo "In ${CFF_PY} (or the _GEN.py that imports it):"
    echo ""
    echo "  SherpackLocation = cms.string('${OUTPUT_FULLPATH}'),"
    echo "  SherpackChecksum = cms.string('${MD5}'),"
    echo ""
    echo "If you move ${OUTPUT} to a different path, update SherpackLocation."
    echo "Previously pointed to: ${INPUT}"
    echo "If SherpaProcess needs editing, check Run.dat in the sherpack."
    echo "=========================================================="

fi

echo ""
echo "Done."
