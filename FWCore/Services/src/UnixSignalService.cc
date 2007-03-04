#include <iostream>
#include <cstdlib>

#include "FWCore/Services/src/UnixSignalService.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

using namespace std;

namespace edm {

  namespace service {

  UnixSignalService::UnixSignalService(edm::ParameterSet const& pset,
                                       edm::ActivityRegistry& registry)
  {
    edm::installCustomHandler(SIGUSR2,edm::ep_sigusr2);
  }

  UnixSignalService::~UnixSignalService() {}

} // end of namespace service
} // end of namespace edm
