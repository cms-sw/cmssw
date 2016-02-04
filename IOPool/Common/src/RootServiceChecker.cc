#include "IOPool/Common/interface/RootServiceChecker.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/RootHandlers.h"

namespace edm {
  RootServiceChecker::RootServiceChecker() {
    Service<RootHandlers> rootSvc;
    if (!rootSvc.isAvailable()) {
      throw edm::Exception(errors::Configuration) <<
	    "The 'InitRootHandlers' service was not specified.\n" <<
	    "This service must be used if PoolSource or PoolOutputModule is used.\n";
    }
  }
}

