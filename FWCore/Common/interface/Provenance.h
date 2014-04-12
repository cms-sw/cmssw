#ifndef FWCore_Common_Provenance_h
#define FWCore_Common_Provenance_h

#include <string>
#include "DataFormats/Provenance/interface/Provenance.h"

namespace edm {
  class ParameterSet;
  std::string moduleName(Provenance const& provenance);
  ParameterSet const& parameterSet(Provenance const& provenance);
}
#endif
