#include "FWCore/Framework/interface/ExternalInputSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
    ExternalInputSource::ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
      ConfigurableInputSource(pset, desc),
      fileNames_(pset.getUntrackedParameter<std::vector<std::string> >("fileNames")) {
    }
    ExternalInputSource::~ExternalInputSource() {}
}
