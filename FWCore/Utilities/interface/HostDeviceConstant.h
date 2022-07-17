#ifndef FWCore_Utilities_HostDeviceConstant_h
#define FWCore_Utilities_HostDeviceConstant_h

// The use of host-side consexpr constants in device code is limited to:
//   - scalars (other than `long double`)
//   - scalar elements of aggregates used inside `constexpr` functions
//
// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-variables .
//
// In particular, it's not possible to use constexpr scalars by pointer or reference (e.g. std::min() takes arguments
// by reference), or pass constexpr arrays as pointers, or access elements of constexpr arrays outside of constexpr
// functions.
//
// The workaround is to define a macro that evaluates to "constexpr" on the host, and "__device__ constexpr" on the
// device. Such macro can be used to declare aggregate objects that are available both on the host and on the device.
// Note these objects may be at different memory addresses on the host and device, so their pointers will be different
// -- but the actual values should be the same.

#ifdef __CUDA_ARCH__
#define HOST_DEVICE_CONSTANT __device__ constexpr
#else
#define HOST_DEVICE_CONSTANT constexpr
#endif

#endif  // FWCore_Utilities_HostDeviceConstant_h
