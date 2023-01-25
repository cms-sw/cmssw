#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ModuleFactory_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ModuleFactory_h

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification
// use the serial_sync variant for cfi file generation with the type@alpaka C++ type
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE2(type_ns, type) \
  DEFINE_FWK_EVENTSETUP_MODULE(type_ns);                    \
  DEFINE_DESC_FILLER_FOR_ESPRODUCERS_IMPL(type_ns, #type "@alpaka", _1)
#define DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(type) \
  DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::type, type)
#else
#define DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE2(type_ns) DEFINE_FWK_EVENTSETUP_MODULE(type_ns)
#define DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(type) \
  DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::type)
#endif

#endif
