#include "FWCore/Sources/interface/ExternalInputSource.h"

namespace edm {
    ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc, bool realData) :
      ConfigurableInputSource(pset, desc, realData),
      poolCatalog_(),
      catalog_(pset, poolCatalog_) {
    }
    ExternalInputSource::~ExternalInputSource() {}
}
