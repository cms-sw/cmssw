if [[ `[ -f /usr/bin/lsb_release ] && lsb_release -si` == "Fedora" ]]
then
  os_arch_comp="el9_amd64_gcc13"
  source /cvmfs/cms.cern.ch/${os_arch_comp}/lcg/root/6.30.07-38879e46537b73c431ecb76409c95eac/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/${os_arch_comp}/external/tbb/v2021.9.0-ef248624754cb25d582754024f93e5e8
  export GCC_CXXCOMPILER_BASE=$(which gcc | sed s[/bin/gcc[[)
elif [[ `[ -f /etc/redhat-release ] && awk -F. '{print $1}' /etc/redhat-release` == "AlmaLinux release 9" ]]
then
  os_arch_comp="el9_amd64_gcc12"
  source /cvmfs/cms.cern.ch/${os_arch_comp}/lcg/root/6.30.09-0308019b8ff223bcf0391fcc4df2b105/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/${os_arch_comp}/external/tbb/v2021.9.0-f55817ca40e6490787280b5021409fd9
  export GCC_CXXCOMPILER_BASE=$(which gcc | sed s[/bin/gcc[[)
elif [[ `[ -f /etc/redhat-release ] && awk -F. '{print $1}' /etc/redhat-release` == "AlmaLinux release 8" ]]
then
  os_arch_comp="el8_amd64_gcc12"
  source /cvmfs/cms.cern.ch/${os_arch_comp}/lcg/root/6.30.09-c59d6b036f1cbd6988c172ba319259f1/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/${os_arch_comp}/external/tbb/v2021.9.0-2391c941213c757dc9a1835b31681235
  export GCC_CXXCOMPILER_BASE=$(which gcc | sed s[/bin/gcc[[)
else
  os_arch_comp="slc7_amd64_gcc12"
  source /cvmfs/cms.cern.ch/${os_arch_comp}/lcg/root/6.30.09-c59d6b036f1cbd6988c172ba319259f1/etc/profile.d/init.sh
  export TBB_GCC=/cvmfs/cms.cern.ch/${os_arch_comp}/external/tbb/v2021.9.0-2391c941213c757dc9a1835b31681235
  export GCC_CXXCOMPILER_BASE=$(which gcc | sed s[/bin/gcc[[)
fi
