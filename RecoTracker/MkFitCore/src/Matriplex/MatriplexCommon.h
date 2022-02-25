#ifndef RecoTracker_MkFitCore_src_Matriplex_MatriplexCommon_h
#define RecoTracker_MkFitCore_src_Matriplex_MatriplexCommon_h

#include <cstring>

// Use intrinsics version of code when available, done via CPP flags.
// #define  MPLEX_USE_INTRINSICS

//==============================================================================
// Intrinsics -- preamble
//==============================================================================

#if defined(__x86_64__)
#include "immintrin.h"
#else
#include <stdlib.h>
#define _mm_malloc(a, b) aligned_alloc(b, a)
#define _mm_free(p) free(p)
#define _mm_prefetch(a, b) __builtin_prefetch(a)
#endif

#if defined(MPLEX_USE_INTRINSICS)
// This seems unnecessary: __AVX__ is usually defined for all higher ISA extensions
#if defined(__AVX__) || defined(__AVX512F__)

#define MPLEX_INTRINSICS

#endif

#if defined(__AVX512F__)

typedef __m512 IntrVec_t;
#define MPLEX_INTRINSICS_WIDTH_BYTES 64
#define MPLEX_INTRINSICS_WIDTH_BITS 512
#define AVX512_INTRINSICS
#define GATHER_INTRINSICS
#define GATHER_IDX_LOAD(name, arr) __m512i name = _mm512_load_epi32(arr);

#define LD(a, i) _mm512_load_ps(&a[i * N + n])
#define ST(a, i, r) _mm512_store_ps(&a[i * N + n], r)
#define ADD(a, b) _mm512_add_ps(a, b)
#define MUL(a, b) _mm512_mul_ps(a, b)
#define FMA(a, b, v) _mm512_fmadd_ps(a, b, v)

#elif defined(__AVX2__) && defined(__FMA__)

typedef __m256 IntrVec_t;
#define MPLEX_INTRINSICS_WIDTH_BYTES 32
#define MPLEX_INTRINSICS_WIDTH_BITS 256
#define AVX2_INTRINSICS
#define GATHER_INTRINSICS
// Previously used _mm256_load_epi32(arr) here, but that's part of AVX-512F, not AVX2
#define GATHER_IDX_LOAD(name, arr) __m256i name = _mm256_load_si256(reinterpret_cast<const __m256i *>(arr));

#define LD(a, i) _mm256_load_ps(&a[i * N + n])
#define ST(a, i, r) _mm256_store_ps(&a[i * N + n], r)
#define ADD(a, b) _mm256_add_ps(a, b)
#define MUL(a, b) _mm256_mul_ps(a, b)
#define FMA(a, b, v) _mm256_fmadd_ps(a, b, v)

#elif defined(__AVX__)

typedef __m256 IntrVec_t;
#define MPLEX_INTRINSICS_WIDTH_BYTES 32
#define MPLEX_INTRINSICS_WIDTH_BITS 256
#define AVX_INTRINSICS

#define LD(a, i) _mm256_load_ps(&a[i * N + n])
#define ST(a, i, r) _mm256_store_ps(&a[i * N + n], r)
#define ADD(a, b) _mm256_add_ps(a, b)
#define MUL(a, b) _mm256_mul_ps(a, b)
// #define FMA(a, b, v)  { __m256 temp = _mm256_mul_ps(a, b); v = _mm256_add_ps(temp, v); }
inline __m256 FMA(const __m256 &a, const __m256 &b, const __m256 &v) {
  __m256 temp = _mm256_mul_ps(a, b);
  return _mm256_add_ps(temp, v);
}

#endif

#endif

#ifdef __INTEL_COMPILER
#define ASSUME_ALIGNED(a, b) __assume_aligned(a, b)
#else
#define ASSUME_ALIGNED(a, b) a = static_cast<decltype(a)>(__builtin_assume_aligned(a, b))
#endif

namespace Matriplex {
  typedef int idx_t;

  void align_check(const char *pref, void *adr);
}  // namespace Matriplex

#endif
