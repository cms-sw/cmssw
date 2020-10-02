#ifndef FWCore_Utilities_GlobalIdentifier_h
#define FWCore_Utilities_GlobalIdentifier_h

#include <string>
namespace edm {
  std::string createGlobalIdentifier(bool binary = false);
}

#endif
