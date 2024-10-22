#ifndef Utilities_GetEnvironmentVariable_h
#define Utilities_GetEnvironmentVariable_h

#include <cstdlib>
#include <string>

namespace edm {
  inline std::string getEnvironmentVariable(std::string const& name, std::string const& defaultValue = std::string()) {
    char* p = std::getenv(name.c_str());
    return (p ? std::string(p) : defaultValue);
  }
}  // namespace edm
#endif
