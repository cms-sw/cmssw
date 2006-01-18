#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
    GeneratedInputSource::GeneratedInputSource(ParameterSet const& pset,
        InputSourceDescription const& desc) :
        ConfigurableInputSource(pset, desc) {
      // maxEvents has already been read in the base class, where it is defaulted.
      // For a generated input source, maxEvents is a required parameter.
      // Here We read it without a default,
      // only so that we will throw if it is not explicitly specified.
      // Therefore, we need not save the value read.
      pset.getUntrackedParameter<int>("maxEvents");
    }
    GeneratedInputSource::~GeneratedInputSource() {}
}
