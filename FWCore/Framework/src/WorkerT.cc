#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm{
  UnscheduledHandler* getUnscheduledHandler(EventPrincipal const& ep) {
    return ep.unscheduledHandler().get();
  }
}
