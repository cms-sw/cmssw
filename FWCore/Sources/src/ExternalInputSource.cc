#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
    ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ConfigurableInputSource(pset, desc),
      catalog_(pset) {
    }
    ExternalInputSource::~ExternalInputSource() {}
}
