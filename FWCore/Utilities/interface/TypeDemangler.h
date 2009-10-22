#ifndef FWCore_Utilities_TypeDemangler_h
#define FWCore_Utilities_TypeDemangler_h

#include <string>

namespace edm {
  void
  typeDemangle(char const* mangledName, std::string& demangledName);
}
#endif
