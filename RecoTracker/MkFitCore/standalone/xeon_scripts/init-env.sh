#!/bin/bash
if [[ `[ -f /usr/bin/lsb_release ] && lsb_release -si` == \
        "Fedora" ||
      `[ -f /etc/redhat-release ] && awk '{print $1}' /etc/redhat-release` == \
        "AlmaLinux" ]]
then
  source /cvmfs/cms.cern.ch/el8_amd64_gcc11/lcg/root/6.26.07-e76d33b57cbf5a1c12b870df2a6ebb79/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/tbb/v2021.8.0-791ebc4967ab49af60ca9ad2aa021259
#  source /cvmfs/cms.cern.ch/el8_amd64_gcc10/lcg/root/6.24.07-3ea108fb48fc7b8bb2960269f724c0d6/etc/profile.d/init.sh
#  export TBB_GCC=/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/tbb/v2021.5.0-36aff7df349e0716374b1668ccd18e17
else
  source /cvmfs/cms.cern.ch/slc7_amd64_gcc11/lcg/root/6.26.07-e76d33b57cbf5a1c12b870df2a6ebb79/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/slc7_amd64_gcc11/external/tbb/v2021.8.0-791ebc4967ab49af60ca9ad2aa021259
#  source /cvmfs/cms.cern.ch/slc7_amd64_gcc10/lcg/root/6.24.07-3ea108fb48fc7b8bb2960269f724c0d6/etc/profile.d/init.sh
#  export TBB_GCC=/cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/tbb/v2021.5.0-36aff7df349e0716374b1668ccd18e17
fi
