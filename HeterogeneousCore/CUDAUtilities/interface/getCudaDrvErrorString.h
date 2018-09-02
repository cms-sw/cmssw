#ifndef HeterogeneousCore_CUDAUtilities_getCudaDrvErrorString_h
#define HeterogeneousCore_CUDAUtilities_getCudaDrvErrorString_h

#include <cuda.h>

inline const char *getCudaDrvErrorString(CUresult error_id) {
  const char *message;
  auto ret = cuGetErrorName(error_id, &message);
  if(ret == CUDA_ERROR_INVALID_VALUE) {
    return static_cast<const char *>("CUDA_ERROR not found!");
  }
  else {
    return message;
  }
}

#endif
