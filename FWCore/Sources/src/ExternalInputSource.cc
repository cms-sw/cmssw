#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
    ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc, bool realData) :
      ConfigurableInputSource(pset, desc, realData),
      catalog_(pset, pset.getUntrackedParameter<std::vector<std::string> >("fileNames")) {
    }
    ExternalInputSource::~ExternalInputSource() {}
}
