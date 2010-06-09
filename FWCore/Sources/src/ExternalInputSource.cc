#include "FWCore/Sources/interface/ExternalInputSource.h"

namespace edm {
    ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc, bool realData) :
      ConfigurableInputSource(pset, desc, realData),
      catalog_(pset) {
    }
    ExternalInputSource::~ExternalInputSource() {}
}
