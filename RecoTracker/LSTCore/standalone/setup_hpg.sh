#!/bin/bash

# HiPerGator module setup for cuda
module load cuda/11.4.3 git
# module use ~/module
# module load root/6.22.08

###########################################################################################################
# Setup environments
###########################################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/code/rooutil/thisrooutil.sh

export SCRAM_ARCH=el8_amd64_gcc12
export CMSSW_VERSION=CMSSW_14_1_0_pre3
export CUDA_HOME=${HPC_CUDA_DIR}

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/$SCRAM_ARCH/cms/cmssw/$CMSSW_VERSION/src
eval `scramv1 runtime -sh`

# Export paths to libraries we need
export BOOST_ROOT=$(scram tool info boost | grep BOOST_BASE | cut -d'=' -f2)
export ALPAKA_ROOT=$(scram tool info alpaka | grep ALPAKA_BASE | cut -d'=' -f2)
export ROOT_ROOT=$(scram tool info root_interface | grep ROOT_INTERFACE_BASE | cut -d'=' -f2)
export ROCM_ROOT=$(scram tool info rocm | grep ROCM_BASE | cut -d'=' -f2)

cd - > /dev/null
echo "Setup following ROOT. Make sure the appropriate setup file has been run. Otherwise the looper won't compile."
which root

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export LD_LIBRARY_PATH=$DIR/SDL/cuda:$DIR/SDL/cpu:$DIR:$LD_LIBRARY_PATH
export PATH=$DIR/bin:$PATH
export PATH=$DIR/efficiency/bin:$PATH
export PATH=$DIR/efficiency/python:$PATH
export TRACKLOOPERDIR=$DIR
export TRACKINGNTUPLEDIR=/blue/p.chang/p.chang/data/lst/CMSSW_12_2_0_pre2
export LSTOUTPUTDIR=.
export LSTPERFORMANCEWEBDIR=/home/users/phchang/public_html/LSTPerformanceWeb

###########################################################################################################
# Validation scripts
###########################################################################################################

# List of benchmark efficiencies are set as an environment variable
export LATEST_CPU_BENCHMARK_EFF_MUONGUN=
export LATEST_CPU_BENCHMARK_EFF_PU200=
#eof
