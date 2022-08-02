// The omission of #include guards is on purpose: it does make sense to #include
// this file multiple times, setting a different value of GPU_DEBUG beforehand.

#ifdef __CUDA_ARCH__
#ifndef GPU_DEBUG

// disable asserts
#ifndef NDEBUG
#define NDEBUG
#endif

#else  // GPU_DEBUG

// enable asserts
#ifdef NDEBUG
#undef NDEBUG
#endif

#endif  // GPU_DEBUG
#endif  // __CUDA_ARCH__

#include <cassert>

#ifdef __CUDA_ARCH__
#ifndef GPU_DEBUG

// replace the no-op assert() with a check and a __trap() instruction
#undef assert
#define assert(expr) \
  do                 \
    if (not(expr)) { \
      __trap();      \
    }                \
  while (false)

#endif  // GPU_DEBUG
#endif  // __CUDA_ARCH__
