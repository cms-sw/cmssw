#include <fstream>
#include <regex>
#include <string>

#include "processor_model.h"

std::string read_processor_model()
{
#ifdef __linux__
  static const std::regex pattern("^model name\\s*:\\s*(.*)", std::regex::optimize);
  std::smatch match;

  std::ifstream cpuinfo("/proc/cpuinfo", std::ios::in);
  std::string line;
  while (cpuinfo.good()) {
    std::getline(cpuinfo, line);
    if (std::regex_match(line, match, pattern)) {
      return match[1];
    }
  }
#endif // __linux__

  return "unknown";
}

const std::string processor_model = read_processor_model();
