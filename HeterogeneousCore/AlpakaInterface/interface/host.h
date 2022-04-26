#ifndef HeterogeneousCore_AlpakaInterface_interface_host_h
#define HeterogeneousCore_AlpakaInterface_interface_host_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace alpaka_common {

  // alpaka host device
  static inline const DevHost host() {
    static const auto host = alpaka::getDevByIdx<alpaka_common::PltfHost>(0u);
    return host;
  }

}  // namespace alpaka_common

#endif  // HeterogeneousCore_AlpakaInterface_interface_host_h
