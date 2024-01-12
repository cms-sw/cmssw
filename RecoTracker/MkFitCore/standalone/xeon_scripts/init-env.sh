#!/bin/bash
if [[ `[ -f /usr/bin/lsb_release ] && lsb_release -si` == "Fedora" ]]
then
  os_arch_comp="el9_amd64_gcc13"
elif [[ `[ -f /etc/redhat-release ] && awk '{print $1}' /etc/redhat-release` == "AlmaLinux" ]]
then
  os_arch_comp="el8_amd64_gcc11"
else
  os_arch_comp="slc7_amd64_gcc11"
fi

source /cvmfs/cms.cern.ch/${os_arch_comp}/lcg/root/6.28.07-573b0d3de9894ea2ab667c0d36cf4882/etc/profile.d/init.sh
export TBB_GCC=/cvmfs/cms.cern.ch/${os_arch_comp}/external/tbb/v2021.9.0-295412b9bb1d6b3275d2ace3e62c1faa
