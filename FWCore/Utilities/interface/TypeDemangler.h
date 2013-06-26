#ifndef FWCore_Utilities_TypeDemangler_h
#define FWCore_Utilities_TypeDemangler_h

#include <string>

namespace edm {
  std::string
  typeDemangle(char const* mangledName);
}
#endif
