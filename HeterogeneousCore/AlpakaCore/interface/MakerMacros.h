#ifndef HeterogeneousCore_AlpakaCore_interface_MakerMacros_h
#define HeterogeneousCore_AlpakaCore_interface_MakerMacros_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// force expanding ALPAKA_ACCELERATOR_NAMESPACE before stringification
// use the serial_sync variant for cfi file generation with the type@alpaka C++ type
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
#define DEFINE_FWK_ALPAKA_MODULE2(name_ns, name) \
  DEFINE_FWK_MODULE(name_ns);                    \
  DEFINE_FWK_PSET_DESC_FILLER_IMPL(name_ns, #name "@alpaka", _1)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name, name)
#else
#define DEFINE_FWK_ALPAKA_MODULE2(name_ns) DEFINE_FWK_MODULE(name_ns)
#define DEFINE_FWK_ALPAKA_MODULE(name) DEFINE_FWK_ALPAKA_MODULE2(ALPAKA_ACCELERATOR_NAMESPACE::name)
#endif

#endif  // HeterogeneousCore_AlpakaCore_interface_MakerMacros_h
