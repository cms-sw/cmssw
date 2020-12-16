#ifndef FWCore_Utilities_interface_CMSUnrollLoop_h
#define FWCore_Utilities_interface_CMSUnrollLoop_h

// convert the macro argument to a null-terminated quoted string
#define STRINGIFY_(ARG) #ARG
#define STRINGIFY(ARG) STRINGIFY_(ARG)

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// CUDA or HIP device compiler

#define CMS_UNROLL_LOOP _Pragma(STRINGIFY(unroll))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(STRINGIFY(unroll N))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(STRINGIFY(unroll 1))

#define CMS_DEVICE_UNROLL_LOOP _Pragma(STRINGIFY(unroll))
#define CMS_DEVICE_UNROLL_LOOP_COUNT(N) _Pragma(STRINGIFY(unroll N))
#define CMS_DEVICE_UNROLL_LOOP_DISABLE _Pragma(STRINGIFY(unroll 1))

#else  // defined (__CUDA_ARCH__) || defined (__HIP_DEVICE_COMPILE__)

// any host compiler
#define CMS_DEVICE_UNROLL_LOOP
#define CMS_DEVICE_UNROLL_LOOP_COUNT(N)
#define CMS_DEVICE_UNROLL_LOOP_DISABLE

#if defined(__clang__)
// clang host compiler

#define CMS_UNROLL_LOOP _Pragma(STRINGIFY(clang loop unroll(enable)))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(STRINGIFY(clang loop unroll_count(N)))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(STRINGIFY(clang loop unroll(disable)))

#elif defined(__GNUC__)
// GCC host compiler

#define CMS_UNROLL_LOOP _Pragma(STRINGIFY(GCC ivdep))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(STRINGIFY(GCC unroll N)) _Pragma(STRINGIFY(GCC ivdep))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(STRINGIFY(GCC unroll 1))

#else
// unsupported or unknown compiler

#define CMS_UNROLL_LOOP
#define CMS_UNROLL_LOOP_COUNT(N)
#define CMS_UNROLL_LOOP_DISABLE

#endif  // defined(__clang__) || defined(__GNUC__) || ...

#endif  // defined (__CUDA_ARCH__) || defined (__HIP_DEVICE_COMPILE__)

#endif  // FWCore_Utilities_interface_CMSUnrollLoop_h
