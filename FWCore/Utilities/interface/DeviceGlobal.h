#ifndef FWCore_Utilities_DeviceGlobal_h
#define FWCore_Utilities_DeviceGlobal_h

// FIXME alpaka provides ALPAKA_STATIC_ACC_MEM_GLOBAL to declare device global
// variables, but it is currently not working as expected. Improve its behaviour
// and syntax and migrate to that.

#if defined(__SYCL_DEVICE_ONLY__)

// The SYCL standard does not support device global variables.
// oneAPI defines the sycl_ext_oneapi_device_global extension, but with an awkward syntax
// that is not easily compatible with CUDA, HIP and regular C++ global variables.
#error "The SYCL backend does not support device global variables"
#define DEVICE_GLOBAL

#elif defined(__CUDA_ARCH__) or defined(__HIP_DEVICE_COMPILE__)

// CUDA and HIP/ROCm device compilers use the __device__ attribute.
#define DEVICE_GLOBAL __device__

#else

// host compilers do not need any special attributes.
#define DEVICE_GLOBAL

#endif

#endif  // FWCore_Utilities_DeviceGlobal_h
