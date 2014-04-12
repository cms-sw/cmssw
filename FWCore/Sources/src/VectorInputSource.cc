/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  VectorInputSource::VectorInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    EDInputSource(pset, desc) {}

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
    this->dropUnwantedBranches_(wantedBranches);
  }

  void
  VectorInputSource::clearEventPrincipal(EventPrincipal& cache) {
    cache.clearEventPrincipal();
  }
}
