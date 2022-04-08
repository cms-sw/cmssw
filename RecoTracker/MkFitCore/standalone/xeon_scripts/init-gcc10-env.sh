#!/bin/bash
source /cvmfs/cms.cern.ch/slc7_amd64_gcc10/lcg/root/6.24.07-f52350f4e0b802edeb9a2551a7d00b92/etc/profile.d/init.sh
export TBB_GCC=/cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/tbb/v2021.4.0-d0152ca29055e3a1bbf629673f6e97c4
# workaround for https://github.com/cms-sw/cmsdist/issues/5574
# remove when we switch to a ROOT build where that issues is fixed
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBJPEG_TURBO_ROOT/lib64
### source /opt/intel/bin/compilervars.sh intel64
