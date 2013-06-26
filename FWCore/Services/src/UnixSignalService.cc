#include <iostream>
#include <cstdlib>

#include "FWCore/Services/src/UnixSignalService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  namespace service {

    UnixSignalService::UnixSignalService(ParameterSet const& pset)
      : enableSigInt_(pset.getUntrackedParameter<bool>("EnableCtrlC")) {
      installCustomHandler(SIGUSR2, ep_sigusr2);
      if(enableSigInt_)  installCustomHandler(SIGINT, ep_sigusr2);
    }

    UnixSignalService::~UnixSignalService() {}

    void UnixSignalService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<bool>("EnableCtrlC", true)->setComment(
        "If 'true', you can stop a cmsRun job gracefully by sending it a '<control> c' keyboard interrupt (i.e. SIGINT).");
      descriptions.add("UnixSignalService", desc);
      descriptions.setComment(
        "This service sets up unix signal handlers for the unix signal SIGUSR2 and optionally SIGINT"
        " so that when cmsRun is sent a signal the application will stop processing and shut down gracefully.");
    }
  } // end of namespace service
} // end of namespace edm
