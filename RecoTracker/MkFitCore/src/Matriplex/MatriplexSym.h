#ifndef RecoTracker_MkFitCore_src_Matriplex_MatriplexSym_h
#define RecoTracker_MkFitCore_src_Matriplex_MatriplexSym_h

#include "MatriplexCommon.h"
#include "Matriplex.h"

//==============================================================================
// MatriplexSym
//==============================================================================

namespace Matriplex {

  const idx_t gSymOffsets[7][36] = {{},
                                    {},
                                    {0, 1, 1, 2},
                                    {0, 1, 3, 1, 2, 4, 3, 4, 5},  // 3
                                    {},
                                    {},
                                    {0, 1, 3, 6, 10, 15, 1,  2,  4,  7,  11, 16, 3,  4,  5,  8,  12, 17,
                                     6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20}};

  //------------------------------------------------------------------------------

  template <typename T, idx_t D, idx_t N>
  class MatriplexSym {
  public:
    typedef T value_type;

    /// no. of matrix rows
    static constexpr int kRows = D;
    /// no. of matrix columns
    static constexpr int kCols = D;
    /// no of elements: lower triangle
    static constexpr int kSize = (D + 1) * D / 2;
    /// size of the whole matriplex
    static constexpr int kTotSize = N * kSize;

    T fArray[kTotSize] __attribute__((aligned(64)));

    MatriplexSym() {}
    MatriplexSym(T v) { setVal(v); }

    idx_t plexSize() const { return N; }

    void setVal(T v) {
      for (idx_t i = 0; i < kTotSize; ++i) {
        fArray[i] = v;
      }
    }

    void add(const MatriplexSym& v) {
      for (idx_t i = 0; i < kTotSize; ++i) {
        fArray[i] += v.fArray[i];
      }
    }

    void scale(T scale) {
      for (idx_t i = 0; i < kTotSize; ++i) {
        fArray[i] *= scale;
      }
    }

    T operator[](idx_t xx) const { return fArray[xx]; }
    T& operator[](idx_t xx) { return fArray[xx]; }

    const idx_t* offsets() const { return gSymOffsets[D]; }
    idx_t off(idx_t i) const { return gSymOffsets[D][i]; }

    const T& constAt(idx_t n, idx_t i, idx_t j) const { return fArray[off(i * D + j) * N + n]; }

    T& At(idx_t n, idx_t i, idx_t j) { return fArray[off(i * D + j) * N + n]; }

    T& operator()(idx_t n, idx_t i, idx_t j) { return At(n, i, j); }
    const T& operator()(idx_t n, idx_t i, idx_t j) const { return constAt(n, i, j); }

    MatriplexSym& operator=(const MatriplexSym& m) {
      memcpy(fArray, m.fArray, sizeof(T) * kTotSize);
      return *this;
    }

    void copySlot(idx_t n, const MatriplexSym& m) {
      for (idx_t i = n; i < kTotSize; i += N) {
        fArray[i] = m.fArray[i];
      }
    }

    void copyIn(idx_t n, const T* arr) {
      for (idx_t i = n; i < kTotSize; i += N) {
        fArray[i] = *(arr++);
      }
    }

    void copyIn(idx_t n, const MatriplexSym& m, idx_t in) {
      for (idx_t i = n; i < kTotSize; i += N, in += N) {
        fArray[i] = m[in];
      }
    }

    void copy(idx_t n, idx_t in) {
      for (idx_t i = n; i < kTotSize; i += N, in += N) {
        fArray[i] = fArray[in];
      }
    }

#if defined(AVX512_INTRINSICS)

    template <typename U>
    void slurpIn(const T* arr, __m512i& vi, const U&, const int N_proc = N) {
      //_mm512_prefetch_i32gather_ps(vi, arr, 1, _MM_HINT_T0);

      const __m512 src = {0};
      const __mmask16 k = N_proc == N ? -1 : (1 << N_proc) - 1;

      for (int i = 0; i < kSize; ++i, ++arr) {
        //_mm512_prefetch_i32gather_ps(vi, arr+2, 1, _MM_HINT_NTA);

        __m512 reg = _mm512_mask_i32gather_ps(src, k, vi, arr, sizeof(U));
        _mm512_mask_store_ps(&fArray[i * N], k, reg);
      }
    }

    // Experimental methods, slurpIn() seems to be at least as fast.
    // See comments in mkFit/MkFitter.cc MkFitter::addBestHit().

    void ChewIn(const char* arr, int off, int vi[N], const char* tmp, __m512i& ui) {
      // This is a hack ... we know sizeof(Hit) = 64 = cache line = vector width.

      for (int i = 0; i < N; ++i) {
        __m512 reg = _mm512_load_ps(arr + vi[i]);
        _mm512_store_ps((void*)(tmp + 64 * i), reg);
      }

      for (int i = 0; i < kSize; ++i) {
        __m512 reg = _mm512_i32gather_ps(ui, tmp + off + i * sizeof(T), 1);
        _mm512_store_ps(&fArray[i * N], reg);
      }
    }

    void Contaginate(const char* arr, int vi[N], const char* tmp) {
      // This is a hack ... we know sizeof(Hit) = 64 = cache line = vector width.

      for (int i = 0; i < N; ++i) {
        __m512 reg = _mm512_load_ps(arr + vi[i]);
        _mm512_store_ps((void*)(tmp + 64 * i), reg);
      }
    }

    void Plexify(const char* tmp, __m512i& ui) {
      for (int i = 0; i < kSize; ++i) {
        __m512 reg = _mm512_i32gather_ps(ui, tmp + i * sizeof(T), 1);
        _mm512_store_ps(&fArray[i * N], reg);
      }
    }

#elif defined(AVX2_INTRINSICS)

    template <typename U>
    void slurpIn(const T* arr, __m256i& vi, const U&, const int N_proc = N) {
      const __m256 src = {0};

      __m256i k = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      __m256i k_sel = _mm256_set1_epi32(N_proc);
      __m256i k_master = _mm256_cmpgt_epi32(k_sel, k);

      k = k_master;
      for (int i = 0; i < kSize; ++i, ++arr) {
        __m256 reg = _mm256_mask_i32gather_ps(src, arr, vi, (__m256)k, sizeof(U));
        // Restore mask (docs say gather clears it but it doesn't seem to).
        k = k_master;
        _mm256_maskstore_ps(&fArray[i * N], k, reg);
      }
    }

#else

    void slurpIn(const T* arr, int vi[N], const int N_proc = N) {
      // Separate N_proc == N case (gains about 7% in fit test).
      if (N_proc == N) {
        for (int i = 0; i < kSize; ++i) {
          for (int j = 0; j < N; ++j) {
            fArray[i * N + j] = *(arr + i + vi[j]);
          }
        }
      } else {
        for (int i = 0; i < kSize; ++i) {
          for (int j = 0; j < N_proc; ++j) {
            fArray[i * N + j] = *(arr + i + vi[j]);
          }
        }
      }
    }

#endif

    void copyOut(idx_t n, T* arr) const {
      for (idx_t i = n; i < kTotSize; i += N) {
        *(arr++) = fArray[i];
      }
    }

    void setDiagonal3x3(idx_t n, T d) {
      T* p = fArray + n;

      p[0 * N] = d;
      p[1 * N] = 0;
      p[2 * N] = d;
      p[3 * N] = 0;
      p[4 * N] = 0;
      p[5 * N] = d;
    }

    MatriplexSym& subtract(const MatriplexSym& a, const MatriplexSym& b) {
      // Does *this = a - b;

#pragma omp simd
      for (idx_t i = 0; i < kTotSize; ++i) {
        fArray[i] = a.fArray[i] - b.fArray[i];
      }

      return *this;
    }

    // ==================================================================
    // Operations specific to Kalman fit in 6 parameter space
    // ==================================================================

    void addNoiseIntoUpperLeft3x3(T noise) {
      T* p = fArray;
      ASSUME_ALIGNED(p, 64);

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
        p[0 * N + n] += noise;
        p[2 * N + n] += noise;
        p[5 * N + n] += noise;
      }
    }

    void invertUpperLeft3x3() {
      typedef T TT;

      T* a = fArray;
      ASSUME_ALIGNED(a, 64);

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
        const TT c00 = a[2 * N + n] * a[5 * N + n] - a[4 * N + n] * a[4 * N + n];
        const TT c01 = a[4 * N + n] * a[3 * N + n] - a[1 * N + n] * a[5 * N + n];
        const TT c02 = a[1 * N + n] * a[4 * N + n] - a[2 * N + n] * a[3 * N + n];
        const TT c11 = a[5 * N + n] * a[0 * N + n] - a[3 * N + n] * a[3 * N + n];
        const TT c12 = a[3 * N + n] * a[1 * N + n] - a[4 * N + n] * a[0 * N + n];
        const TT c22 = a[0 * N + n] * a[2 * N + n] - a[1 * N + n] * a[1 * N + n];

        // Force determinant calculation in double precision.
        const double det = (double)a[0 * N + n] * c00 + (double)a[1 * N + n] * c01 + (double)a[3 * N + n] * c02;
        const TT s = TT(1) / det;

        a[0 * N + n] = s * c00;
        a[1 * N + n] = s * c01;
        a[2 * N + n] = s * c11;
        a[3 * N + n] = s * c02;
        a[4 * N + n] = s * c12;
        a[5 * N + n] = s * c22;
      }
    }
  };

  template <typename T, idx_t D, idx_t N>
  using MPlexSym = MatriplexSym<T, D, N>;

  //==============================================================================
  // Multiplications
  //==============================================================================

  template <typename T, idx_t D, idx_t N>
  struct SymMultiplyCls {
    static void multiply(const MPlexSym<T, D, N>& A, const MPlexSym<T, D, N>& B, MPlex<T, D, D, N>& C) {
      throw std::runtime_error("general symmetric multiplication not supported");
    }
  };

  template <typename T, idx_t N>
  struct SymMultiplyCls<T, 3, N> {
    static void multiply(const MPlexSym<T, 3, N>& A, const MPlexSym<T, 3, N>& B, MPlex<T, 3, 3, N>& C) {
      const T* a = A.fArray;
      ASSUME_ALIGNED(a, 64);
      const T* b = B.fArray;
      ASSUME_ALIGNED(b, 64);
      T* c = C.fArray;
      ASSUME_ALIGNED(c, 64);

#ifdef MPLEX_INTRINSICS

      for (idx_t n = 0; n < N; n += 64 / sizeof(T)) {
#include "intr_sym_3x3.ah"
      }

#else

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
#include "std_sym_3x3.ah"
      }

#endif
    }
  };

  template <typename T, idx_t N>
  struct SymMultiplyCls<T, 6, N> {
    static void multiply(const MPlexSym<float, 6, N>& A, const MPlexSym<float, 6, N>& B, MPlex<float, 6, 6, N>& C) {
      const T* a = A.fArray;
      ASSUME_ALIGNED(a, 64);
      const T* b = B.fArray;
      ASSUME_ALIGNED(b, 64);
      T* c = C.fArray;
      ASSUME_ALIGNED(c, 64);

#ifdef MPLEX_INTRINSICS

      for (idx_t n = 0; n < N; n += 64 / sizeof(T)) {
#include "intr_sym_6x6.ah"
      }

#else

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
#include "std_sym_6x6.ah"
      }

#endif
    }
  };

  template <typename T, idx_t D, idx_t N>
  void multiply(const MPlexSym<T, D, N>& A, const MPlexSym<T, D, N>& B, MPlex<T, D, D, N>& C) {
    SymMultiplyCls<T, D, N>::multiply(A, B, C);
  }

  //==============================================================================
  // Cramer inversion
  //==============================================================================

  template <typename T, idx_t D, idx_t N>
  struct CramerInverterSym {
    static void invert(MPlexSym<T, D, N>& A, double* determ = nullptr) {
      throw std::runtime_error("general cramer inversion not supported");
    }
  };

  template <typename T, idx_t N>
  struct CramerInverterSym<T, 2, N> {
    static void invert(MPlexSym<T, 2, N>& A, double* determ = nullptr) {
      typedef T TT;

      T* a = A.fArray;
      ASSUME_ALIGNED(a, 64);

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
        // Force determinant calculation in double precision.
        const double det = (double)a[0 * N + n] * a[2 * N + n] - (double)a[1 * N + n] * a[1 * N + n];
        if (determ)
          determ[n] = det;

        const TT s = TT(1) / det;
        const TT tmp = s * a[2 * N + n];
        a[1 * N + n] *= -s;
        a[2 * N + n] = s * a[0 * N + n];
        a[0 * N + n] = tmp;
      }
    }
  };

  template <typename T, idx_t N>
  struct CramerInverterSym<T, 3, N> {
    static void invert(MPlexSym<T, 3, N>& A, double* determ = nullptr) {
      typedef T TT;

      T* a = A.fArray;
      ASSUME_ALIGNED(a, 64);

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
        const TT c00 = a[2 * N + n] * a[5 * N + n] - a[4 * N + n] * a[4 * N + n];
        const TT c01 = a[4 * N + n] * a[3 * N + n] - a[1 * N + n] * a[5 * N + n];
        const TT c02 = a[1 * N + n] * a[4 * N + n] - a[2 * N + n] * a[3 * N + n];
        const TT c11 = a[5 * N + n] * a[0 * N + n] - a[3 * N + n] * a[3 * N + n];
        const TT c12 = a[3 * N + n] * a[1 * N + n] - a[4 * N + n] * a[0 * N + n];
        const TT c22 = a[0 * N + n] * a[2 * N + n] - a[1 * N + n] * a[1 * N + n];

        // Force determinant calculation in double precision.
        const double det = (double)a[0 * N + n] * c00 + (double)a[1 * N + n] * c01 + (double)a[3 * N + n] * c02;
        if (determ)
          determ[n] = det;

        const TT s = TT(1) / det;
        a[0 * N + n] = s * c00;
        a[1 * N + n] = s * c01;
        a[2 * N + n] = s * c11;
        a[3 * N + n] = s * c02;
        a[4 * N + n] = s * c12;
        a[5 * N + n] = s * c22;
      }
    }
  };

  template <typename T, idx_t D, idx_t N>
  void invertCramerSym(MPlexSym<T, D, N>& A, double* determ = nullptr) {
    CramerInverterSym<T, D, N>::invert(A, determ);
  }

  //==============================================================================
  // Cholesky inversion
  //==============================================================================

  template <typename T, idx_t D, idx_t N>
  struct CholeskyInverterSym {
    static void invert(MPlexSym<T, D, N>& A) { throw std::runtime_error("general cholesky inversion not supported"); }
  };

  template <typename T, idx_t N>
  struct CholeskyInverterSym<T, 3, N> {
    static void invert(MPlexSym<T, 3, N>& A) {
      typedef T TT;

      T* a = A.fArray;

#pragma omp simd
      for (idx_t n = 0; n < N; ++n) {
        TT l0 = std::sqrt(T(1) / a[0 * N + n]);
        TT l1 = a[1 * N + n] * l0;
        TT l2 = a[2 * N + n] - l1 * l1;
        l2 = std::sqrt(T(1) / l2);
        TT l3 = a[3 * N + n] * l0;
        TT l4 = (a[4 * N + n] - l1 * l3) * l2;
        TT l5 = a[5 * N + n] - (l3 * l3 + l4 * l4);
        l5 = std::sqrt(T(1) / l5);

        // decomposition done

        l3 = (l1 * l4 * l2 - l3) * l0 * l5;
        l1 = -l1 * l0 * l2;
        l4 = -l4 * l2 * l5;

        a[0 * N + n] = l3 * l3 + l1 * l1 + l0 * l0;
        a[1 * N + n] = l3 * l4 + l1 * l2;
        a[2 * N + n] = l4 * l4 + l2 * l2;
        a[3 * N + n] = l3 * l5;
        a[4 * N + n] = l4 * l5;
        a[5 * N + n] = l5 * l5;

        // m(2,x) are all zero if anything went wrong at l5.
        // all zero, if anything went wrong already for l0 or l2.
      }
    }
  };

  template <typename T, idx_t D, idx_t N>
  void invertCholeskySym(MPlexSym<T, D, N>& A) {
    CholeskyInverterSym<T, D, N>::invert(A);
  }

}  // end namespace Matriplex

#endif
