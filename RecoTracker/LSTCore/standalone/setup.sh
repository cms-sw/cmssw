#!/bin/bash

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh

ARCH=$(uname -m)
if [ -z ${CMSSW_SEARCH_PATH+x} ]; then
  if [ -z ${FORCED_CMSSW_VERSION+x} ]; then
    export CMSSW_VERSION=CMSSW_14_2_0_pre4
  else
    export CMSSW_VERSION=$FORCED_CMSSW_VERSION
  fi

  source /cvmfs/cms.cern.ch/cmsset_default.sh
  CMSSW_PATH=$(scram list -c CMSSW | grep -w $CMSSW_VERSION | awk '{print $3}')
  cd $CMSSW_PATH
  eval `scramv1 runtime -sh`
else
  cd $CMSSW_BASE/src
fi

# Export paths to libraries we need
export BOOST_ROOT=$(scram tool info boost | grep BOOST_BASE | cut -d'=' -f2)
export ALPAKA_ROOT=$(scram tool info alpaka | grep ALPAKA_BASE | cut -d'=' -f2)
export CUDA_HOME=$(scram tool info cuda | grep CUDA_BASE | cut -d'=' -f2)
export ROOT_ROOT=$(scram tool info root_interface | grep ROOT_INTERFACE_BASE | cut -d'=' -f2)
export ROCM_ROOT=$(scram tool info rocm | grep ROCM_BASE | cut -d'=' -f2)

cd - > /dev/null
echo "Setup following ROOT. Make sure the appropriate setup file has been run. Otherwise the looper won't compile."
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR/LST:$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
export TRACKINGNTUPLEDIR=/data2/segmentlinking/CMSSW_12_2_0_pre2/
export LSTOUTPUTDIR=.

hostname=$(hostname)
if [[ $hostname == *cornell* ]]; then
  export LSTPERFORMANCEWEBDIR="/cdat/tem/${USER}/LSTPerformanceWeb"
else
  export LSTPERFORMANCEWEBDIR="/home/users/phchang/public_html/LSTPerformanceWeb"
fi

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN="/data2/segmentlinking/muonGun_cpu_efficiencies.root"
export LATEST_CPU_BENCHMARK_EFF_PU200="/data2/segmentlinking/pu200_cpu_efficiencies.root"
#eof
