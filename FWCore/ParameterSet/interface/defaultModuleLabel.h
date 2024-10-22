#ifndef FWCore_ParameterSet_defaultModuleLabel_h
#define FWCore_ParameterSet_defaultModuleLabel_h

#include <string>

namespace edm {
  // take by value because we'll copy it anyway
  std::string defaultModuleLabel(std::string label);
}  // namespace edm

#endif
