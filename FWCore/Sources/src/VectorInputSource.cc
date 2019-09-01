/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Sources/interface/VectorInputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {

  struct VectorInputSourceDescription;

  VectorInputSource::VectorInputSource(ParameterSet const& pset, VectorInputSourceDescription const& desc)
      : productRegistry_(desc.productRegistry_),
        processHistoryRegistry_(new ProcessHistoryRegistry),
        consecutiveRejectionsLimit_(pset.getUntrackedParameter<unsigned int>("consecutiveRejectionsLimit", 100)) {}

  VectorInputSource::~VectorInputSource() {}

  void VectorInputSource::dropUnwantedBranches(std::vector<std::string> const& wantedBranches) {
    this->dropUnwantedBranches_(wantedBranches);
  }

  void VectorInputSource::clearEventPrincipal(EventPrincipal& cache) { cache.clearEventPrincipal(); }

  void VectorInputSource::doBeginJob() { this->beginJob(); }

  void VectorInputSource::doEndJob() { this->endJob(); }

  void VectorInputSource::throwIfOverLimit(unsigned int consecutiveRejections) const {
    if (consecutiveRejections >= consecutiveRejectionsLimit_) {
      throw cms::Exception("LogicError")
          << "VectorInputSource::loopOverEvents() read " << consecutiveRejections
          << " consecutive pileup events that were rejected by the eventOperator. "
          << "This is likely a sign of misconfiguration (of e.g. the adjusted-to pileup probability profile). "
          << "If you know what you're doing, this exception can be turned off by setting consecutiveRejectionsLimit=0.";
    }
  }

}  // namespace edm
