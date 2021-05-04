#ifndef HeterogeneousCore_CUDAServices_numberOfDevices_h
#define HeterogeneousCore_CUDAServices_numberOfDevices_h

namespace cms {
  namespace cuda {
    // Returns the number of CUDA devices
    // The difference wrt. the standard CUDA function or
    // cms::cuda::deviceCount() is that if CUDAService is disabled,
    // this function returns 0.
    int numberOfDevices();
  }  // namespace cuda
}  // namespace cms

#endif
