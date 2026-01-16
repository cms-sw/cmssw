#ifndef HeterogeneousCore_AlpakaInterface_interface_atomicMaxF_h
#define HeterogeneousCore_AlpakaInterface_interface_atomicMaxF_h

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/bit_cast.h"

// FIXME: this should be rewritten using the correct template specialisation for the different accelerator types

template <alpaka::concepts::Acc TAcc>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static float atomicMaxF(const TAcc& acc, float* address, float val) {
#if defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)
  // GPU implementation uses __float_as_int / __int_as_float
  int ret = __float_as_int(*address);
  while (val > __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int*)address, old, __float_as_int(val))) == old)
      break;
  }
  return __int_as_float(ret);
#else
  // CPU implementation uses edm::bit_cast
  int ret = edm::bit_cast<int>(*address);
  while (val > edm::bit_cast<float>(ret)) {
    int old = ret;
    if ((ret = alpaka::atomicCas(acc, (int*)address, old, edm::bit_cast<int>(val))) == old)
      break;
  }
  return edm::bit_cast<float>(ret);
#endif  // __CUDA_ARCH__ or __HIP_DEVICE_COMPILE__
}

#endif  // HeterogeneousCore_AlpakaInterface_interface_atomicMaxF_h
