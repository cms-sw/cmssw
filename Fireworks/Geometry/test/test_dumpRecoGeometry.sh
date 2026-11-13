#!/bin/bash
function die { echo $1: status $2 ; exit $2; }

# Define the base directory where the geometry files are located
GEOMETRY_DIR="${CMSSW_BASE}/src/Configuration/Geometry/python"
if [ ! -e ${GEOMETRY_DIR} ] ; then
  GEOMETRY_DIR="${CMSSW_RELEASE_BASE}/src/Configuration/Geometry/python"
fi

# Check if the TAG is provided as an argument
if [ -z "$1" ]; then
  echo "Error: No TAG provided. Usage: ./script.sh <TAG>"
  exit 1
fi

# Get the TAG from the command-line argument
TAG=$1

# Function to extract the available versions for Run4.
# Only versions that have a Reco geometry file AND are registered in
# upgradeProperties['Run4'] (which provides the Era and GlobalTag) are used.
get_Run4_versions() {
  local files=($(ls ${GEOMETRY_DIR}/GeometryExtendedRun4D*Reco*))
  if [ ${#files[@]} -eq 0 ]; then
    echo "No files found for Run4 versions." >&2
    exit 1
  fi

  local registered
  registered=$(python3 -c "
from Configuration.PyReleaseValidation.upgradeWorkflowComponents import upgradeProperties as p
print(' '.join(k for k in p['Run4']))") || { echo "Failed to read upgradeProperties" >&2; exit 1; }

  local versions=()
  for file in "${files[@]}"; do
    local version=$(basename "$file" | sed -n 's/.*GeometryExtendedRun4D\([0-9]\{1,3\}\).*/\1/p')
    if [[ "$version" =~ ^[0-9]{1,3}$ ]]; then
      if [[ " $registered " == *" Run4D${version} "* ]]; then
        versions+=("D${version}")
      else
        echo "Skipping D${version}: not registered in upgradeProperties['Run4']" >&2
      fi
    fi
  done

  # Return the unique sorted list of versions
  echo "${versions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '
}

# Set the number of parallel jobs (adjust based on system resources)
MAX_JOBS=4

# PIDs and versions of the background cmsRun jobs
PIDS=()
PID_VERSIONS=()

# Function to run cmsRun and limit parallel jobs
run_cmsrun() {
  local tag=$1
  local version=$2

  cmsRun $CMSSW_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=$tag version=$version &
  PIDS+=($!)
  PID_VERSIONS+=("$version")

  # Wait for jobs to finish if the number of background jobs reaches MAX_JOBS
  while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
    sleep 1
  done
}

# Check if the tag is Run4
if [ "$TAG" == "Run4" ]; then
  # Get all the versions for Run4
  VERSIONS=($(get_Run4_versions)) || exit 1
  if [ ${#VERSIONS[@]} -eq 0 ]; then
    echo "No registered Run4 versions found."
    exit 1
  fi
  for VERSION in "${VERSIONS[@]}"; do
    echo "Running for Run4 with version $VERSION"
    run_cmsrun "Run4" "$VERSION"
  done

  # Wait for all background jobs to finish and collect their exit codes
  FAILED=()
  STATUS=0
  for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    rc=$?
    if [ $rc -ne 0 ]; then
      FAILED+=("${PID_VERSIONS[$i]}")
      STATUS=$rc
    fi
  done
  if [ ${#FAILED[@]} -ne 0 ]; then
    die "Failure running dumpRecoGeometry_cfg.py tag=$TAG for version(s): ${FAILED[*]}" $STATUS
  fi
else
  echo "Running for tag $TAG"
  cmsRun $CMSSW_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=$TAG || die "Failure running dumpRecoGeometry_cfg.py tag=$TAG" $?
fi
