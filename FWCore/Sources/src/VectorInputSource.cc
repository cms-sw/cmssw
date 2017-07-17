/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  struct VectorInputSourceDescription;

  VectorInputSource::VectorInputSource(ParameterSet const& pset, VectorInputSourceDescription const& desc) : 
      productRegistry_(desc.productRegistry_),
      processHistoryRegistry_(new ProcessHistoryRegistry) {
  }

  VectorInputSource::~VectorInputSource() {}

  void
  VectorInputSource::dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
    this->dropUnwantedBranches_(wantedBranches);
  }

  void
  VectorInputSource::clearEventPrincipal(EventPrincipal& cache) {
    cache.clearEventPrincipal();
  }

  void
  VectorInputSource::doBeginJob() {
    this->beginJob();
  }

  void
  VectorInputSource::doEndJob() {
    this->endJob();
  }

}
