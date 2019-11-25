#ifndef FWCore_Common_Provenance_h
#define FWCore_Common_Provenance_h

#include <string>
#include "DataFormats/Provenance/interface/Provenance.h"

namespace edm {
  class ParameterSet;
  class ProcessHistory;
  std::string moduleName(Provenance const& provenance, ProcessHistory const& history);
  ParameterSet const& parameterSet(Provenance const& provenance, ProcessHistory const& history);
}  // namespace edm
#endif
