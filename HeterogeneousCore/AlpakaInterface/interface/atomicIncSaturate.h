#ifndef HeterogeneousCore_AlpakaInterface_interface_atomicIncSaturate_h
#define HeterogeneousCore_AlpakaInterface_interface_atomicIncSaturate_h

#include <alpaka/alpaka.hpp>

// This function is similar to atomicInc, but instead of wrapping around it saturates at the given value.

ALPAKA_NO_HOST_ACC_WARNING
template <typename TAcc, typename T, typename THierarchy = alpaka::hierarchy::Grids>
ALPAKA_FN_HOST_ACC auto atomicIncSaturate(TAcc const& acc,
                                          T* address,
                                          T const& limit,
                                          THierarchy const& hierarchy = THierarchy()) -> T {
  T assumed;
  T old = *address;

  do {
    assumed = old;
    if (assumed >= limit) {
      // Saturate at limit.
      break;
    }
    old = alpaka::atomicCas(acc, address, assumed, assumed + 1, hierarchy);
  } while (old != assumed);

  return old;
}

#endif  // HeterogeneousCore_AlpakaInterface_interface_atomicIncSaturate_h
