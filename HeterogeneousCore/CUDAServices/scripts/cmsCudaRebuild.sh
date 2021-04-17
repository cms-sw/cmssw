#! /bin/bash -e

function help() {
  cat <<@EOF
Usage:
  cmsCudaRebuild.sh [-g|-G] [-v] [-h]

Check out and build all packages that contain CUDA code in .cu files.

Options:
  -g    Compile with debugging symbols, passing
          "-g -rdynamic" to the host compiler, and
          "-g -lineinfo" to CUDA compiler

  -G    Compile with debugging symbols and enable asserts on the GPU, passing
          "-g -rdynamic -DGPU_DEBUG" to the host compiler, and
          "-g -lineinfo -DGPU_DEBUG" to the CUDA compiler.

  -h    Show this help, and exit.

  -v    Make scram be verbose.

@EOF
}


DEBUG=0
VERBOSE=0

while [ "$*" ]; do
  case "$1" in
  -h)
    help
    exit 0
    ;;
  -g)
    DEBUG=1
    shift
    ;;
  -G)
    DEBUG=2
    shift
    ;;
  -v)
    VERBOSE=$((VERBOSE + 1))
    shift
    ;;
  *)
    help
    exit 1
    ;;
  esac
done

# move to the .../src directory
cd $CMSSW_BASE/src/

# check out all packages containing .cu files
git ls-files --full-name | grep '.*\.cu$' | cut -d/ -f-2 | sort -u | xargs git cms-addpkg

# set additional compilation flags
if (( DEBUG == 1 )); then
  export USER_CXXFLAGS="-g -rdynamic $USER_CXXFLAGS"
  export USER_CUDA_FLAGS="-g -lineinfo $USER_CUDA_FLAGS"
elif (( DEBUG == 2 )); then
  export USER_CXXFLAGS="-g -rdynamic -DGPU_DEBUG $USER_CXXFLAGS"
  export USER_CUDA_FLAGS="-g -lineinfo -DGPU_DEBUG $USER_CUDA_FLAGS"
fi

if (( VERBOSE > 0 )); then
  SCRAM_VERBOSE="-v"
fi

# clean all built packages
scram b clean

# rebuild all checked out packages
scram b $SCRAM_VERBOSE -j
