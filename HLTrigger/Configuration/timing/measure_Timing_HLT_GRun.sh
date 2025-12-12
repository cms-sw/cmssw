#! /bin/bash -e

SOURCE=/data/store/data/Run2024E/EphemeralHLTPhysics/FED/run381065_cff.py

# check for the source configuration
if ! [ -f $SOURCE ]; then
  echo "The necessary input data cannot be found, the test will be skipped"
  exit 0
fi

# link the source configuration locally
ln -sf $SOURCE source_cff.py

# check for the HLT configuration
if ! [ -f Timing_HLT_GRun.py ]; then
  echo "Cannot find the HLT configuration, the test will be skipped"
  exit 1
fi

# check out the timing scripts, if needed
if ! [ -d "patatrack-scripts" ]; then
  git clone https://github.com/cms-patatrack/patatrack-scripts.git
  if ! [ -x "patatrack-scripts/benchmark" ]; then
    echo "Cannot find the timing scripts, the test will be skipped"
    exit 0
  fi
  echo
fi

# start a local instance of the NVIDIA MPS server
CUDA_MPS_PATH=$(mktemp -d)
mkdir -p $CUDA_MPS_PATH/{control,log}
export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PATH/control
export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_PATH/log
nvidia-cuda-mps-control -d

# benchmark the HLT execution time
patatrack-scripts/benchmark -r 4 -e 20300 -j 8 -t 32 -s 24 -g 1 --numa-affinity --no-cpu-affinity -l logs -k resources.json -- Timing_HLT_GRun.py
mergeResourcesJson.py logs/step*/pid*/resources.json > resources.json

# stop the local instance of the NVIDIA MPS server
echo quit | nvidia-cuda-mps-control
rm -rf $CUDA_MPS_PATH
