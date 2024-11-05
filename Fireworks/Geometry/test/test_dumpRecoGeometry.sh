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

# Function to extract the available versions for Run4
get_Run4_versions() {
  local files=($(ls ${GEOMETRY_DIR}/GeometryExtendedRun4D*Reco*))
  if [ ${#files[@]} -eq 0 ]; then
    echo "No files found for Run4 versions."
    exit 1
  fi

  local versions=()
  for file in "${files[@]}"; do
    local version=$(basename "$file" | sed -n 's/.*GeometryExtendedRun4D\([0-9]\{1,3\}\).*/\1/p')
    if [[ "$version" =~ ^[0-9]{1,3}$ ]]; then
      versions+=("D${version}")
    fi
  done

  # Return the unique sorted list of versions
  echo "${versions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '
}

# Set the number of parallel jobs (adjust based on system resources)
MAX_JOBS=4

# Function to run cmsRun and limit parallel jobs
run_cmsrun() {
  local tag=$1
  local version=$2

  cmsRun $CMSSW_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=$tag version=$version &

  # Wait for jobs to finish if the number of background jobs reaches MAX_JOBS
  while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
    sleep 1
  done
}

# Check if the tag is Run4
if [ "$TAG" == "Run4" ]; then
  # Get all the versions for Run4
  VERSIONS=($(get_Run4_versions))
  for VERSION in "${VERSIONS[@]}"; do
    echo "Running for Run4 with version $VERSION"
    run_cmsrun "Run4" "$VERSION"  || die "Failure running dumpRecoGeometry_cfg.py tag=$TAG" $?
  done

  # Wait for all background jobs to finish
  wait
else
  echo "Running for tag $TAG"
  cmsRun $CMSSW_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=$TAG || die "Failure running dumpRecoGeometry_cfg.py tag=$TAG" $?
  if [ $? -ne 0 ]; then
    echo "Error running cmsRun for tag=$TAG"
    exit 1
  fi
fi
