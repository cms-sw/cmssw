#include <iostream>
#include <cstdlib>

#include "FWCore/Services/src/UnixSignalService.h"

#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

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
