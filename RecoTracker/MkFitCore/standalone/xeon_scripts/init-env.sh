#!/bin/bash
if [[ `lsb_release -si` == "Fedora" ]]
then
  source /cvmfs/cms.cern.ch/el8_amd64_gcc10/lcg/root/6.24.07-da610b2b7ed663a0a05d3605f3d83ceb/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/tbb/v2021.5.0-e966a5acb1e4d5fd7605074bafbb079c/
else
  source /cvmfs/cms.cern.ch/slc7_amd64_gcc10/lcg/root/6.24.07-f52350f4e0b802edeb9a2551a7d00b92/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/tbb/v2021.4.0-d0152ca29055e3a1bbf629673f6e97c4
fi
