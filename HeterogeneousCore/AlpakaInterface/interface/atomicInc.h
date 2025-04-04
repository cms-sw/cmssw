#ifndef HeterogeneousCore_AlpakaInterface_interface_atomicInc_h
#define HeterogeneousCore_AlpakaInterface_interface_atomicInc_h

#include <alpaka/alpaka.hpp>

// This function is similar to atomicInc, but deduces the limiting value from the type itself.

ALPAKA_NO_HOST_ACC_WARNING
template <typename TAcc, typename T, typename THierarchy = alpaka::hierarchy::Grids>
ALPAKA_FN_HOST_ACC auto atomicInc(TAcc const& acc, T* address, THierarchy const& hierarchy = THierarchy()) -> T {
  T limit = std::numeric_limits<T>::max();
  return alpaka::atomicInc(acc, address, limit, hierarchy);
}

#endif  // HeterogeneousCore_AlpakaInterface_interface_atomicInc_h
