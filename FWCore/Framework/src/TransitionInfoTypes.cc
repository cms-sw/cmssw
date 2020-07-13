#include "FWCore/Framework/src/TransitionInfoTypes.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"

namespace edm {

  EventSetupImpl const& ProcessBlockTransitionInfo::eventSetupImpl() const {
    static const EventSetupImpl dummyEventSetupImpl;
    return dummyEventSetupImpl;
  }

}  // namespace edm
