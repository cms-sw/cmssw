#!/bin/bash
source /cvmfs/cms.cern.ch/slc7_amd64_gcc10/lcg/root/6.20.06-cms/etc/profile.d/init.sh
export TBB_GCC=/cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/tbb/2020_U2
# workaround for https://github.com/cms-sw/cmsdist/issues/5574
# remove when we switch to a ROOT build where that issues is fixed
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBJPEG_TURBO_ROOT/lib64
### source /opt/intel/bin/compilervars.sh intel64
