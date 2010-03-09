#include <iostream>
#include <cstdlib>

#include "FWCore/Services/src/UnixSignalService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  namespace service {

  UnixSignalService::UnixSignalService(edm::ParameterSet const& pset,
                                       edm::ActivityRegistry& registry)
    : enableSigInt_(pset.getUntrackedParameter<bool>("EnableCtrlC"))
  {
    edm::installCustomHandler(SIGUSR2,edm::ep_sigusr2);
    if(enableSigInt_)  edm::installCustomHandler(SIGINT ,edm::ep_sigusr2);
  }

  UnixSignalService::~UnixSignalService() {}

  void UnixSignalService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("EnableCtrlC",true);
    descriptions.add("UnixSignalService", desc);
  }

} // end of namespace service
} // end of namespace edm
