#!/bin/bash
if [[ `[ -f /usr/bin/lsb_release ] && lsb_release -si` == \
        "Fedora" ||
      `[ -f /etc/redhat-release ] && awk '{print $1}' /etc/redhat-release` == \
        "AlmaLinux" ]]
then
  os_arch_comp="el8_amd64_gcc11"
else
  os_arch_comp="slc7_amd64_gcc11"
fi
  source /cvmfs/cms.cern.ch/${os_arch_comp}/lcg/root/6.26.07-e76d33b57cbf5a1c12b870df2a6ebb79/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/${os_arch_comp}/external/tbb/v2021.8.0-791ebc4967ab49af60ca9ad2aa021259
