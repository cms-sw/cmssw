#!/bin/bash -x
function die { echo $1: status $2 ; exit $2; }

# Define the base directory where the geometry files are located
GEOMETRY_DIR="${CMSSW_BASE}/src/Configuration/Geometry/python"
if [ ! -e ${GEOMETRY_DIR} ] ; then
  GEOMETRY_DIR="${CMSSW_RELEASE_BASE}/src/Configuration/Geometry/python"
fi

# Define the list of tags
TAGS=("Run1" "2015" "2017" "2026") #"2021"

# Function to extract the available versions for 2026
get_2026_versions() {
  local files=($(ls ${GEOMETRY_DIR}/GeometryExtended2026D*Reco*))
  if [ ${#files[@]} -eq 0 ]; then
    echo "No files found for 2026 versions."
    exit 1
  fi

  local versions=()
  for file in "${files[@]}"; do
    local version=$(basename "$file" | sed -n 's/.*GeometryExtended2026D\([0-9]\{1,3\}\).*/\1/p')
    if [[ "$version" =~ ^[0-9]{1,3}$ ]]; then
      versions+=("D${version}")
    fi
  done

  # Return the unique sorted list of versions
  echo "${versions[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '
}

# Iterate over each tag
for TAG in "${TAGS[@]}"; do
  if [ "$TAG" == "2026" ]; then
    # Get all the versions for 2026
    VERSIONS=($(get_2026_versions))
    for VERSION in "${VERSIONS[@]}"; do
      echo "Running for 2026 with version $VERSION"
      cmsRun $CMSSW_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=2026 version=$VERSION
      # Check the exit status of the command
      if [ $? -ne 0 ]; then
        echo "Error running cmsRun for tag=2026 and version=$VERSION"
        exit 1
      fi
    done
  else
    echo "Running for tag $TAG"
    cmsRun $CMSSW_BASE/src/Fireworks/Geometry/python/dumpRecoGeometry_cfg.py tag=$TAG
    # Check the exit status of the command
    if [ $? -ne 0 ]; then
      echo "Error running cmsRun for tag=$TAG"
      exit 1
    fi
  fi
done
