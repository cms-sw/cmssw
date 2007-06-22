#include "FWCore/Sources/interface/ExternalInputSource.h"

namespace edm {
    ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ConfigurableInputSource(pset, desc, true),
      catalog_(pset) {
    }
    ExternalInputSource::~ExternalInputSource() {}
}
