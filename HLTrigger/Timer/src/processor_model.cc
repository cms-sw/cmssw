#include <string>
#include <boost/predef/os.h>

#if BOOST_OS_LINUX
// Linux
#include <fstream>
#include <regex>
#endif  // BOOST_OS_LINUX

#if BOOST_OS_BSD || BOOST_OS_MACOS
// OSX or BSD
#include <sys/types.h>
#include <sys/sysctl.h>
#endif  // BOOST_OS_BSD || BOOST_OS_MACOS

#include "HLTrigger/Timer/interface/processor_model.h"

std::string read_processor_model() {
  std::string result = "unknown";

#if BOOST_OS_LINUX
  // on Linux, read the processor model from /proc/cpuinfo,
  // and check the status of SMT from /sys/devices/system/cpu/smt/active
  static const std::regex pattern("^model name\\s*:\\s*(.*)", std::regex::optimize);
  std::smatch match;

  std::ifstream cpuinfo("/proc/cpuinfo", std::ios::in);
  std::string line;
  while (cpuinfo.good()) {
    std::getline(cpuinfo, line);
    if (std::regex_match(line, match, pattern)) {
      result = match[1];
      break;
    }
  }

  std::ifstream smtinfo("/sys/devices/system/cpu/smt/active", std::ios::in);
  if (smtinfo) {
    int status;
    smtinfo >> status;
    result += status ? " with SMT enabled" : " with SMT disabled";
  }
#endif  // BOOST_OS_LINUX

#if BOOST_OS_BSD || BOOST_OS_MACOS
  // on BSD and OS X, read the processor  model via sysctlbyname("machdep.cpu.brand_string", ...)
  std::string result;
  size_t len;
  sysctlbyname("machdep.cpu.brand_string", nullptr, &len, NULL, 0);
  result.resize(len);
  sysctlbyname("machdep.cpu.brand_string", result.data(), &len, NULL, 0);
#endif  // BOOST_OS_BSD || BOOST_OS_MACOS

  return result;
}

const std::string processor_model = read_processor_model();
