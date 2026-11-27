#ifndef RecoTracker_MkFitCore_src_Matriplex_MatriplexCommon_h
#define RecoTracker_MkFitCore_src_Matriplex_MatriplexCommon_h

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

// Use intrinsics version of code when available, done via CPP flags.
// #define  MPLEX_USE_INTRINSICS

//==============================================================================
// Intrinsics -- preamble
//==============================================================================

#if defined(__x86_64__)
#include "immintrin.h"
#else
#include <cstdlib>
#endif

#ifndef MPLEX_ALIGN
#if defined(__AVX512F__)
#define MPLEX_ALIGN 64
#elif defined(__AVX__) || defined(__AVX2__)
#define MPLEX_ALIGN 32
#elif defined(__SSE3__)
#define MPLEX_ALIGN 16
#else
#define MPLEX_ALIGN 32
#endif
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

  namespace internal {
    template <typename T>
    void sincos4(const T x, T &sin, T &cos) {
      // Had this writen with explicit division by factorial.
      // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

      const T x2 = x * x;
      cos = T(1.0) - T(0.5) * x2 + T(0.0416666666666666667) * x2 * x2;
      sin = x - T(0.166666666666666667) * x * x2;
    }

    // atan2 without branches, so that a loop over it vectorises; vdt::fast_atan2f has branches and does not.
    // t = min(|x|, |y|) / max(|x|, |y|) in [0, 1], atan t = t + t^3 P(t^2) with P of degree 7, then the octant as
    // base + sign atan t, with base 0, pi/2 or pi in two parts. Largest error 1.74 ulp in strict mode, 2.6 ulp with
    // -Ofast and FMA, 3.8 ulp with CMSSW's flags, where the vector division is approximate; vdt::fast_atan2f has
    // 3.2, 3.5 and 3.5 ulp.
    inline float atan2_poly(float y, float x) {
      constexpr float kPio2Hi = 0x1.921fb6p0f, kPio2Lo = -0x1.777a5cp-25f, kPiHi = 0x1.921fb6p1f,
                      kPiLo = -0x1.777a5cp-24f;
      const float ax = std::fabs(x), ay = std::fabs(y);
      const float mx = std::max(ax, ay), mn = std::min(ax, ay);
      const float t = mn / (mx == 0 ? 1.f : mx), t2 = t * t;  // (0, 0): 0
      const float p = -0x1.5554dcp-2f +
                      t2 * (0x1.9978f4p-3f +
                            t2 * (-0x1.230adcp-3f +
                                  t2 * (0x1.b4e12ap-4f +
                                        t2 * (-0x1.3556b6p-4f +
                                              t2 * (0x1.61fddap-5f + t2 * (-0x1.0c2c14p-6f + t2 * 0x1.7ed232p-9f))))));
      float a = t + t * t2 * p;
      const bool swap = ay > ax, xneg = std::signbit(x);
      const float bhi = swap ? kPio2Hi : (xneg ? kPiHi : 0.f), blo = swap ? kPio2Lo : (xneg ? kPiLo : 0.f);
      a = (blo + (swap != xneg ? -a : a)) + bhi;
      return std::copysign(a, y);
    }
  }  // namespace internal
}  // namespace Matriplex

#endif
