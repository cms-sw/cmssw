#ifndef HeterogeneousCore_AlpakaInterface_interface_debug_h
#define HeterogeneousCore_AlpakaInterface_interface_debug_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::debug {

  [[nodiscard]] ALPAKA_FN_ACC inline bool always_true() {
    volatile const bool flag = true;
    return flag;
  }

  [[nodiscard]] ALPAKA_FN_ACC inline bool always_false() {
    volatile const bool flag = false;
    return flag;
  }

  ALPAKA_FN_ACC inline void do_not_optimise(const auto& arg) {
    volatile const auto* ptr = &arg;
    (void)*ptr;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::debug

#endif  // HeterogeneousCore_AlpakaInterface_interface_debug_h
