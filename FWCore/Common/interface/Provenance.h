#ifndef FWCore_Common_Provenance_h
#define FWCore_Common_Provenance_h

#include <string>
#include "DataFormats/Provenance/interface/StableProvenance.h"

namespace edm {
  class ParameterSet;
  class ProcessHistory;
  std::string moduleName(StableProvenance const& provenance, ProcessHistory const& history);
  ParameterSet const& parameterSet(StableProvenance const& provenance, ProcessHistory const& history);
}  // namespace edm
#endif
