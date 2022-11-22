#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ModuleFactory_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ModuleFactory_h

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification inside DEFINE_FWK_EVENTSETUP_MODULE
#define DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE2(type) DEFINE_FWK_EVENTSETUP_MODULE(type)
#define DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(type) \
  DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::type)

#endif
