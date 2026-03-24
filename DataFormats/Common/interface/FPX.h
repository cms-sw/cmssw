#ifndef FPX_h
#define FPX_h

#include <limits>

#if defined ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

/*
 * Portable floating-point precision abstraction for CPU/GPU execution.
 *
 * This header defines a unified floating-point type `FPX` that maps to:
 *  - `__half` (FP16) when compiled with CUDA GPU support
 *  - `float` (FP32) otherwise
 *
 * It is designed to exploit half-precision acceleration on GPUs.
 */

#if defined ALPAKA_ACC_GPU_CUDA_ENABLED
/*
    * GPU (CUDA) implementation:
    * Uses native IEEE-754 half precision (`__half`) type.
    * Provides explicit conversion helpers between float and half.
    */
#define __F2H __float2half
#define __H2F __half2float
typedef __half FPX;

__host__ __device__ inline FPX makeNaN() { return __float2half(std::numeric_limits<float>::quiet_NaN()); }

#else
/*
    * CPU fallback implementation:
    * Uses standard 32-bit floating point arithmetic.
    */
#define __F2H
#define __H2F
typedef float FPX;

inline FPX makeNaN() { return std::numeric_limits<FPX>::quiet_NaN(); }

#endif

#endif
