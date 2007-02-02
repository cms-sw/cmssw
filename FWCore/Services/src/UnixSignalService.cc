#include <iostream>
#include <cstdlib>
#include <signal.h>

#include "FWCore/Services/src/UnixSignalService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

using namespace std;

namespace edm {

  namespace service {

  UnixSignalService::UnixSignalService(edm::ParameterSet const& pset,
                                       edm::ActivityRegistry& registry)
  {
// Establish the handler (ep_sigusr2) for SIGUSR2
    sigset_t oldset;
    edm::disableAllSigs(&oldset);
#if defined(__linux__)
    edm::disableRTSigs();
#endif
    edm::installSig(SIGUSR2,edm::ep_sigusr2);
    edm::reenableSigs(&oldset);

//  registry.watchPostProcessEvent(this,&UnixSignalService::postEventProcessing);
  }

  UnixSignalService::~UnixSignalService() {}

#ifdef NOT_YET
  void UnixSignalService::postEventProcessing( const edm::Event& ev, const edm::EventSetup& es )
  {
    std::cerr << "UnixSignalService::postEventProcessing reached" << std::endl;
  }
#endif
} // end of namespace service
} // end of namespace edm
