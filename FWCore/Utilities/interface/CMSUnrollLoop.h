#ifndef FWCore_Utilities_interface_CMSUnrollLoop_h
#define FWCore_Utilities_interface_CMSUnrollLoop_h

#include "FWCore/Utilities/interface/stringize.h"

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// CUDA or HIP device compiler

#define CMS_UNROLL_LOOP _Pragma(EDM_STRINGIZE(unroll))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(EDM_STRINGIZE(unroll N))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(EDM_STRINGIZE(unroll 1))

#define CMS_DEVICE_UNROLL_LOOP _Pragma(EDM_STRINGIZE(unroll))
#define CMS_DEVICE_UNROLL_LOOP_COUNT(N) _Pragma(EDM_STRINGIZE(unroll N))
#define CMS_DEVICE_UNROLL_LOOP_DISABLE _Pragma(EDM_STRINGIZE(unroll 1))

#else  // defined (__CUDA_ARCH__) || defined (__HIP_DEVICE_COMPILE__)

// any host compiler
#define CMS_DEVICE_UNROLL_LOOP
#define CMS_DEVICE_UNROLL_LOOP_COUNT(N)
#define CMS_DEVICE_UNROLL_LOOP_DISABLE

#if defined(__clang__)
// clang host compiler

#define CMS_UNROLL_LOOP _Pragma(EDM_STRINGIZE(clang loop unroll(enable)))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(EDM_STRINGIZE(clang loop unroll_count(N)))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(EDM_STRINGIZE(clang loop unroll(disable)))

#elif defined(__INTEL_COMPILER)
// Intel icc compiler
#define CMS_UNROLL_LOOP _Pragma(EDM_STRINGIZE(unroll))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(EDM_STRINGIZE(unroll(N)))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(EDM_STRINGIZE(nounroll))

#elif defined(__GNUC__)
// GCC host compiler

#define CMS_UNROLL_LOOP _Pragma(EDM_STRINGIZE(GCC ivdep))
#define CMS_UNROLL_LOOP_COUNT(N) _Pragma(EDM_STRINGIZE(GCC unroll N)) _Pragma(EDM_STRINGIZE(GCC ivdep))
#define CMS_UNROLL_LOOP_DISABLE _Pragma(EDM_STRINGIZE(GCC unroll 1))

#else
// unsupported or unknown compiler

#define CMS_UNROLL_LOOP
#define CMS_UNROLL_LOOP_COUNT(N)
#define CMS_UNROLL_LOOP_DISABLE

#endif  // defined(__clang__) || defined(__GNUC__) || ...

#endif  // defined (__CUDA_ARCH__) || defined (__HIP_DEVICE_COMPILE__)

#endif  // FWCore_Utilities_interface_CMSUnrollLoop_h
