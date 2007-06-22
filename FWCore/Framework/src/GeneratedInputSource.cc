#include "FWCore/Framework/interface/GeneratedInputSource.h"

namespace edm {
    GeneratedInputSource::GeneratedInputSource(ParameterSet const& pset,
        InputSourceDescription const& desc) :
        ConfigurableInputSource(pset, desc, false) {
    }
    GeneratedInputSource::~GeneratedInputSource() {}
}
