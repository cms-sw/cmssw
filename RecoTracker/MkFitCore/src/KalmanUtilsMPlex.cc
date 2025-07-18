#include "KalmanUtilsMPlex.h"
#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

#include "KalmanUtilsMPlex.icc"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"

namespace {
  using namespace mkfit;
  using idx_t = Matriplex::idx_t;

  inline void MultResidualsAdd(const MPlexLH& A, const MPlexLV& B, const MPlex2V& C, MPlexLV& D) {
    // outPar = psPar + kalmanGain*(dPar)
    //   D    =   B         A         C
    // where right half of kalman gain is 0

    // XXX Regenerate with a script.

    MultResidualsAdd_imp(A, B, C, D, 0, NN);
  }

  inline void MultResidualsAdd(const MPlexL2& A, const MPlexLV& B, const MPlex2V& C, MPlexLV& D) {
    // outPar = psPar + kalmanGain*(dPar)
    //   D    =   B         A         C

    // XXX Regenerate with a script.

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    const T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);
    T* d = D.fArray;
    ASSUME_ALIGNED(d, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      // generate loop (can also write it manually this time, it's not much)
      d[0 * N + n] = b[0 * N + n] + a[0 * N + n] * c[0 * N + n] + a[1 * N + n] * c[1 * N + n];
      d[1 * N + n] = b[1 * N + n] + a[2 * N + n] * c[0 * N + n] + a[3 * N + n] * c[1 * N + n];
      d[2 * N + n] = b[2 * N + n] + a[4 * N + n] * c[0 * N + n] + a[5 * N + n] * c[1 * N + n];
      d[3 * N + n] = b[3 * N + n] + a[6 * N + n] * c[0 * N + n] + a[7 * N + n] * c[1 * N + n];
      d[4 * N + n] = b[4 * N + n] + a[8 * N + n] * c[0 * N + n] + a[9 * N + n] * c[1 * N + n];
      d[5 * N + n] = b[5 * N + n] + a[10 * N + n] * c[0 * N + n] + a[11 * N + n] * c[1 * N + n];
    }
  }

  inline void MultResidualsAdd(const MPlex52& A, const MPlex5V& B, const MPlex2V& C, MPlex5V& D) {
    // outPar = psPar + kalmanGain*(dPar)
    //   D    =   B         A         C

    // XXX Regenerate with a script.

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    const T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);
    T* d = D.fArray;
    ASSUME_ALIGNED(d, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      // generate loop (can also write it manually this time, it's not much)
      d[0 * N + n] = b[0 * N + n] + a[0 * N + n] * c[0 * N + n] + a[1 * N + n] * c[1 * N + n];
      d[1 * N + n] = b[1 * N + n] + a[2 * N + n] * c[0 * N + n] + a[3 * N + n] * c[1 * N + n];
      d[2 * N + n] = b[2 * N + n] + a[4 * N + n] * c[0 * N + n] + a[5 * N + n] * c[1 * N + n];
      d[3 * N + n] = b[3 * N + n] + a[6 * N + n] * c[0 * N + n] + a[7 * N + n] * c[1 * N + n];
      d[4 * N + n] = b[4 * N + n] + a[8 * N + n] * c[0 * N + n] + a[9 * N + n] * c[1 * N + n];
    }
  }

  //------------------------------------------------------------------------------

  inline void Chi2Similarity(const MPlex2V& A,  //resPar
                             const MPlex2S& C,  //resErr
                             MPlexQF& D)        //outChi2
  {
    // outChi2 = (resPar) * resErr * (resPar)
    //   D     =    A      *    C   *      A

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);
    T* d = D.fArray;
    ASSUME_ALIGNED(d, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      // generate loop (can also write it manually this time, it's not much)
      d[0 * N + n] = c[0 * N + n] * a[0 * N + n] * a[0 * N + n] + c[2 * N + n] * a[1 * N + n] * a[1 * N + n] +
                     2 * (c[1 * N + n] * a[1 * N + n] * a[0 * N + n]);
    }
  }

  //------------------------------------------------------------------------------

  inline void AddIntoUpperLeft3x3(const MPlexLS& A, const MPlexHS& B, MPlexHS& C) {
    // The rest of matrix is left untouched.

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] + b[0 * N + n];
      c[1 * N + n] = a[1 * N + n] + b[1 * N + n];
      c[2 * N + n] = a[2 * N + n] + b[2 * N + n];
      c[3 * N + n] = a[3 * N + n] + b[3 * N + n];
      c[4 * N + n] = a[4 * N + n] + b[4 * N + n];
      c[5 * N + n] = a[5 * N + n] + b[5 * N + n];
    }
  }

  inline void AddIntoUpperLeft2x2(const MPlexLS& A, const MPlexHS& B, MPlex2S& C) {
    // The rest of matrix is left untouched.

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] + b[0 * N + n];
      c[1 * N + n] = a[1 * N + n] + b[1 * N + n];
      c[2 * N + n] = a[2 * N + n] + b[2 * N + n];
    }
  }

  //------------------------------------------------------------------------------

  inline void SubtractFirst3(const MPlexHV& A, const MPlexLV& B, MPlexHV& C) {
    // The rest of matrix is left untouched.

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] - b[0 * N + n];
      c[1 * N + n] = a[1 * N + n] - b[1 * N + n];
      c[2 * N + n] = a[2 * N + n] - b[2 * N + n];
    }
  }

  inline void SubtractFirst2(const MPlexHV& A, const MPlexLV& B, MPlex2V& C) {
    // The rest of matrix is left untouched.

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (idx_t n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] - b[0 * N + n];
      c[1 * N + n] = a[1 * N + n] - b[1 * N + n];
    }
  }

  //==============================================================================

  inline void ProjectResErr(const MPlexQF& A00, const MPlexQF& A01, const MPlexHS& B, MPlexHH& C) {
    // C = A * B, C is 3x3, A is 3x3 , B is 3x3 sym

    // Based on script generation and adapted to custom sizes.

    typedef float T;
    const idx_t N = NN;

    const T* a00 = A00.fArray;
    ASSUME_ALIGNED(a00, 64);
    const T* a01 = A01.fArray;
    ASSUME_ALIGNED(a01, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a00[n] * b[0 * N + n] + a01[n] * b[1 * N + n];
      c[1 * N + n] = a00[n] * b[1 * N + n] + a01[n] * b[2 * N + n];
      c[2 * N + n] = a00[n] * b[3 * N + n] + a01[n] * b[4 * N + n];
      c[3 * N + n] = b[3 * N + n];
      c[4 * N + n] = b[4 * N + n];
      c[5 * N + n] = b[5 * N + n];
      c[6 * N + n] = a01[n] * b[0 * N + n] - a00[n] * b[1 * N + n];
      c[7 * N + n] = a01[n] * b[1 * N + n] - a00[n] * b[2 * N + n];
      c[8 * N + n] = a01[n] * b[3 * N + n] - a00[n] * b[4 * N + n];
    }
  }

  inline void ProjectResErrTransp(const MPlexQF& A00, const MPlexQF& A01, const MPlexHH& B, MPlex2S& C) {
    // C = A * B, C is 3x3 sym, A is 3x3 , B is 3x3

    // Based on script generation and adapted to custom sizes.

    typedef float T;
    const idx_t N = NN;

    const T* a00 = A00.fArray;
    ASSUME_ALIGNED(a00, 64);
    const T* a01 = A01.fArray;
    ASSUME_ALIGNED(a01, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = b[0 * N + n] * a00[n] + b[1 * N + n] * a01[n];
      c[1 * N + n] = b[3 * N + n] * a00[n] + b[4 * N + n] * a01[n];
      c[2 * N + n] = b[5 * N + n];
    }
  }

  inline void RotateResidualsOnTangentPlane(const MPlexQF& R00,  //r00
                                            const MPlexQF& R01,  //r01
                                            const MPlexHV& A,    //res_glo
                                            MPlex2V& B)          //res_loc
  {
    RotateResidualsOnTangentPlane_impl(R00, R01, A, B, 0, NN);
  }

  //==============================================================================

  /*
  inline void ProjectResErr(const MPlex2H& A, const MPlexHS& B, MPlex2H& C) {
    // C = A * B, C is 2x3, A is 2x3 , B is 3x3 sym

    //
    // A 0 1 2
    //   3 4 5
    // B 0 1 3
    //   1 2 4
    //   3 4 5
    //

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[1 * N + n] + a[2 * N + n] * b[3 * N + n];
      c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[2 * N + n] + a[2 * N + n] * b[4 * N + n];
      c[2 * N + n] = a[0 * N + n] * b[3 * N + n] + a[1 * N + n] * b[4 * N + n] + a[2 * N + n] * b[5 * N + n];
      c[3 * N + n] = a[3 * N + n] * b[0 * N + n] + a[4 * N + n] * b[1 * N + n] + a[5 * N + n] * b[3 * N + n];
      c[4 * N + n] = a[3 * N + n] * b[1 * N + n] + a[4 * N + n] * b[2 * N + n] + a[5 * N + n] * b[4 * N + n];
      c[5 * N + n] = a[3 * N + n] * b[3 * N + n] + a[4 * N + n] * b[4 * N + n] + a[5 * N + n] * b[5 * N + n];
    }
  }
  */

  // inline void ProjectResErr(const MPlex2H& A, const MPlexLS& B, MPlex2H& C) {
  template <class T1, class T2>
  inline void ProjectResErr(const T1& A, const T2& B, MPlex2H& C) {
    // C = A * B, C is 2x3, A is 2x3 , B is 3x3 sym

    /*
    A 0 1 2
      3 4 5
    B 0 1 3
      1 2 4
      3 4 5
    */

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[1 * N + n] + a[2 * N + n] * b[3 * N + n];
      c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[2 * N + n] + a[2 * N + n] * b[4 * N + n];
      c[2 * N + n] = a[0 * N + n] * b[3 * N + n] + a[1 * N + n] * b[4 * N + n] + a[2 * N + n] * b[5 * N + n];
      c[3 * N + n] = a[3 * N + n] * b[0 * N + n] + a[4 * N + n] * b[1 * N + n] + a[5 * N + n] * b[3 * N + n];
      c[4 * N + n] = a[3 * N + n] * b[1 * N + n] + a[4 * N + n] * b[2 * N + n] + a[5 * N + n] * b[4 * N + n];
      c[5 * N + n] = a[3 * N + n] * b[3 * N + n] + a[4 * N + n] * b[4 * N + n] + a[5 * N + n] * b[5 * N + n];
    }
  }

  //inline void ProjectResErrTransp(const MPlex2H& A, const MPlex2H& B, MPlex2S& C) {
  template <class T1>
  inline void ProjectResErrTransp(const T1& A, const MPlex2H& B, MPlex2S& C) {
    // C = B * A^T, C is 2x2 sym, A is 2x3 (A^T is 3x2), B is 2x3

    /*
    B   0 1 2
        3 4 5
    A^T 0 3
        1 4
        2 5
    */

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = b[0 * N + n] * a[0 * N + n] + b[1 * N + n] * a[1 * N + n] + b[2 * N + n] * a[2 * N + n];
      c[1 * N + n] = b[0 * N + n] * a[3 * N + n] + b[1 * N + n] * a[4 * N + n] + b[2 * N + n] * a[5 * N + n];
      c[2 * N + n] = b[3 * N + n] * a[3 * N + n] + b[4 * N + n] * a[4 * N + n] + b[5 * N + n] * a[5 * N + n];
    }
  }

  inline void RotateVectorOnPlane(const MPlexHH& R, const MPlexHV& A, MPlexHV& B) {
    // typedef float T;
    // const idx_t N = NN;

    // const T* a = A.fArray;
    // ASSUME_ALIGNED(a, 64);
    // T* b = B.fArray;
    // ASSUME_ALIGNED(b, 64);
    // const T* r = R.fArray;
    // ASSUME_ALIGNED(r, 64);

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      B(n, 0, 0) = R(n, 0, 0) * A(n, 0, 0) + R(n, 0, 1) * A(n, 1, 0) + R(n, 0, 2) * A(n, 2, 0);
      B(n, 1, 0) = R(n, 1, 0) * A(n, 0, 0) + R(n, 1, 1) * A(n, 1, 0) + R(n, 1, 2) * A(n, 2, 0);
      B(n, 2, 0) = R(n, 2, 0) * A(n, 0, 0) + R(n, 2, 1) * A(n, 1, 0) + R(n, 2, 2) * A(n, 2, 0);
    }
  }

  inline void RotateVectorOnPlaneTransp(const MPlexHH& R, const MPlexHV& A, MPlexHV& B) {
    // typedef float T;
    // const idx_t N = NN;

    // const T* a = A.fArray;
    // ASSUME_ALIGNED(a, 64);
    // T* b = B.fArray;
    // ASSUME_ALIGNED(b, 64);
    // const T* r = R.fArray;
    // ASSUME_ALIGNED(r, 64);

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      B(n, 0, 0) = R(n, 0, 0) * A(n, 0, 0) + R(n, 1, 0) * A(n, 1, 0) + R(n, 2, 0) * A(n, 2, 0);
      B(n, 1, 0) = R(n, 0, 1) * A(n, 0, 0) + R(n, 1, 1) * A(n, 1, 0) + R(n, 2, 1) * A(n, 2, 0);
      B(n, 2, 0) = R(n, 0, 2) * A(n, 0, 0) + R(n, 1, 2) * A(n, 1, 0) + R(n, 2, 2) * A(n, 2, 0);
    }
  }

  template <typename T1, typename T2>
  void RotateResidualsOnPlane(const T1& R,  //prj     - at least MPlex_2_3
                              const T2& A,  //res_glo - at least MPlex_3_1 (vector)
                              MPlex2V& B)   //res_loc - fixed as MPlex_2_1 (vector)
  {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      B(n, 0, 0) = R(n, 0, 0) * A(n, 0, 0) + R(n, 0, 1) * A(n, 1, 0) + R(n, 0, 2) * A(n, 2, 0);
      B(n, 1, 0) = R(n, 1, 0) * A(n, 0, 0) + R(n, 1, 1) * A(n, 1, 0) + R(n, 1, 2) * A(n, 2, 0);
    }
  }

  inline void KalmanHTG(const MPlexQF& A00, const MPlexQF& A01, const MPlex2S& B, MPlexHH& C) {
    // HTG  = rot * res_loc
    //   C  =  A  *    B

    // Based on script generation and adapted to custom sizes.

    typedef float T;
    const idx_t N = NN;

    const T* a00 = A00.fArray;
    ASSUME_ALIGNED(a00, 64);
    const T* a01 = A01.fArray;
    ASSUME_ALIGNED(a01, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a00[n] * b[0 * N + n];
      c[1 * N + n] = a00[n] * b[1 * N + n];
      c[2 * N + n] = 0.;
      c[3 * N + n] = a01[n] * b[0 * N + n];
      c[4 * N + n] = a01[n] * b[1 * N + n];
      c[5 * N + n] = 0.;
      c[6 * N + n] = b[1 * N + n];
      c[7 * N + n] = b[2 * N + n];
      c[8 * N + n] = 0.;
    }
  }

  inline void KalmanGain(const MPlexLS& A, const MPlexHH& B, MPlexLH& C) {
    // C = A * B, C is 6x3, A is 6x6 sym , B is 3x3

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[3 * N + n] + a[3 * N + n] * b[6 * N + n];
      c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[4 * N + n] + a[3 * N + n] * b[7 * N + n];
      c[2 * N + n] = 0;
      c[3 * N + n] = a[1 * N + n] * b[0 * N + n] + a[2 * N + n] * b[3 * N + n] + a[4 * N + n] * b[6 * N + n];
      c[4 * N + n] = a[1 * N + n] * b[1 * N + n] + a[2 * N + n] * b[4 * N + n] + a[4 * N + n] * b[7 * N + n];
      c[5 * N + n] = 0;
      c[6 * N + n] = a[3 * N + n] * b[0 * N + n] + a[4 * N + n] * b[3 * N + n] + a[5 * N + n] * b[6 * N + n];
      c[7 * N + n] = a[3 * N + n] * b[1 * N + n] + a[4 * N + n] * b[4 * N + n] + a[5 * N + n] * b[7 * N + n];
      c[8 * N + n] = 0;
      c[9 * N + n] = a[6 * N + n] * b[0 * N + n] + a[7 * N + n] * b[3 * N + n] + a[8 * N + n] * b[6 * N + n];
      c[10 * N + n] = a[6 * N + n] * b[1 * N + n] + a[7 * N + n] * b[4 * N + n] + a[8 * N + n] * b[7 * N + n];
      c[11 * N + n] = 0;
      c[12 * N + n] = a[10 * N + n] * b[0 * N + n] + a[11 * N + n] * b[3 * N + n] + a[12 * N + n] * b[6 * N + n];
      c[13 * N + n] = a[10 * N + n] * b[1 * N + n] + a[11 * N + n] * b[4 * N + n] + a[12 * N + n] * b[7 * N + n];
      c[14 * N + n] = 0;
      c[15 * N + n] = a[15 * N + n] * b[0 * N + n] + a[16 * N + n] * b[3 * N + n] + a[17 * N + n] * b[6 * N + n];
      c[16 * N + n] = a[15 * N + n] * b[1 * N + n] + a[16 * N + n] * b[4 * N + n] + a[17 * N + n] * b[7 * N + n];
      c[17 * N + n] = 0;
    }
  }

  inline void KalmanHTG(const MPlex2H& A, const MPlex2S& B, MPlexH2& C) {
    // HTG  = prj^T * res_loc
    //   C  =  A^T  *   B

    /*
    A^T 0 3
        1 4
        2 5
    B 0 1
      1 2
    C 0 1
      2 3
      4 5
    */

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[3 * N + n] * b[1 * N + n];
      c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[3 * N + n] * b[2 * N + n];
      c[2 * N + n] = a[1 * N + n] * b[0 * N + n] + a[4 * N + n] * b[1 * N + n];
      c[3 * N + n] = a[1 * N + n] * b[1 * N + n] + a[4 * N + n] * b[2 * N + n];
      c[4 * N + n] = a[2 * N + n] * b[0 * N + n] + a[5 * N + n] * b[1 * N + n];
      c[5 * N + n] = a[2 * N + n] * b[1 * N + n] + a[5 * N + n] * b[2 * N + n];
    }
  }

  inline void KalmanGain(const MPlexLS& A, const MPlexH2& B, MPlexL2& C) {
    // C = A * B, C is 6x2, A is 6x6 sym , B is 3x2 (6x2 but half of it is zeros)

    /*
      A 0  1  3  6 10 15
        1  2  4  7 11 16
        3  4  5  8 12 17
        6  7  8  9 13 18
       10 11 12 13 14 19
       15 16 17 18 19 20
      B 0  1
        2  3
	4  5
        X  X with X=0, so not even included in B
        X  X
        X  X
      C 0  1
        2  3
	4  5
        6  7
        8  9
       10 11
     */

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[2 * N + n] + a[3 * N + n] * b[4 * N + n];
      c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[3 * N + n] + a[3 * N + n] * b[5 * N + n];
      c[2 * N + n] = a[1 * N + n] * b[0 * N + n] + a[2 * N + n] * b[2 * N + n] + a[4 * N + n] * b[4 * N + n];
      c[3 * N + n] = a[1 * N + n] * b[1 * N + n] + a[2 * N + n] * b[3 * N + n] + a[4 * N + n] * b[5 * N + n];
      c[4 * N + n] = a[3 * N + n] * b[0 * N + n] + a[4 * N + n] * b[2 * N + n] + a[5 * N + n] * b[4 * N + n];
      c[5 * N + n] = a[3 * N + n] * b[1 * N + n] + a[4 * N + n] * b[3 * N + n] + a[5 * N + n] * b[5 * N + n];
      c[6 * N + n] = a[6 * N + n] * b[0 * N + n] + a[7 * N + n] * b[2 * N + n] + a[8 * N + n] * b[4 * N + n];
      c[7 * N + n] = a[6 * N + n] * b[1 * N + n] + a[7 * N + n] * b[3 * N + n] + a[8 * N + n] * b[5 * N + n];
      c[8 * N + n] = a[10 * N + n] * b[0 * N + n] + a[11 * N + n] * b[2 * N + n] + a[12 * N + n] * b[4 * N + n];
      c[9 * N + n] = a[10 * N + n] * b[1 * N + n] + a[11 * N + n] * b[3 * N + n] + a[12 * N + n] * b[5 * N + n];
      c[10 * N + n] = a[15 * N + n] * b[0 * N + n] + a[16 * N + n] * b[2 * N + n] + a[17 * N + n] * b[4 * N + n];
      c[11 * N + n] = a[15 * N + n] * b[1 * N + n] + a[16 * N + n] * b[3 * N + n] + a[17 * N + n] * b[5 * N + n];
    }
  }

  inline void CovXYconstrain(const MPlexQF& R00, const MPlexQF& R01, const MPlexLS& Ci, MPlexLS& Co) {
    // C is transformed to align along y after rotation and rotated back

    typedef float T;
    const idx_t N = NN;

    const T* r00 = R00.fArray;
    ASSUME_ALIGNED(r00, 64);
    const T* r01 = R01.fArray;
    ASSUME_ALIGNED(r01, 64);
    const T* ci = Ci.fArray;
    ASSUME_ALIGNED(ci, 64);
    T* co = Co.fArray;
    ASSUME_ALIGNED(co, 64);

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      // a bit loopy to avoid temporaries
      co[0 * N + n] =
          r00[n] * r00[n] * ci[0 * N + n] + 2 * r00[n] * r01[n] * ci[1 * N + n] + r01[n] * r01[n] * ci[2 * N + n];
      co[1 * N + n] = r00[n] * r01[n] * co[0 * N + n];
      co[2 * N + n] = r01[n] * r01[n] * co[0 * N + n];
      co[0 * N + n] = r00[n] * r00[n] * co[0 * N + n];

      co[3 * N + n] = r00[n] * ci[3 * N + n] + r01[n] * ci[4 * N + n];
      co[4 * N + n] = r01[n] * co[3 * N + n];
      co[3 * N + n] = r00[n] * co[3 * N + n];

      co[6 * N + n] = r00[n] * ci[6 * N + n] + r01[n] * ci[7 * N + n];
      co[7 * N + n] = r01[n] * co[6 * N + n];
      co[6 * N + n] = r00[n] * co[6 * N + n];

      co[10 * N + n] = r00[n] * ci[10 * N + n] + r01[n] * ci[11 * N + n];
      co[11 * N + n] = r01[n] * co[10 * N + n];
      co[10 * N + n] = r00[n] * co[10 * N + n];

      co[15 * N + n] = r00[n] * ci[15 * N + n] + r01[n] * ci[16 * N + n];
      co[16 * N + n] = r01[n] * co[15 * N + n];
      co[15 * N + n] = r00[n] * co[15 * N + n];
    }
  }

  void KalmanGain(const MPlexLS& A, const MPlex2S& B, MPlexL2& C) {
    // C = A * B, C is 6x2, A is 6x6 sym , B is 2x2

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "KalmanGain62.ah"
  }

  inline void KHMult(const MPlexLH& A, const MPlexQF& B00, const MPlexQF& B01, MPlexLL& C) {
    // C = A * B, C is 6x6, A is 6x3 , B is 3x6
    KHMult_imp(A, B00, B01, C, 0, NN);
  }

  inline void KHMult(const MPlexL2& A, const MPlex2H& B, MPlexLL& C) {
    // C = A * B, C is 6x6, A is 6x2 , B is 2x3 (2x6 but half of it made of zeros)

    /*
    A 0  1
      2  3
      4  5
      6  7
      8  9
     10 11
    B  0  1  2  X  X  X with X=0 so not included in B
       3  4  5  X  X  X
    C  0  1  2  3  4  5
       6  7  8  9 10 11
      12 13 14 15 16 17
      18 19 20 21 22 23
      24 25 26 27 28 29
      30 31 32 33 34 34
    */

    // typedef float T;
    // const idx_t N = NN;

    // const T* a = A.fArray;
    // ASSUME_ALIGNED(a, 64);
    // const T* b = B.fArray;
    // ASSUME_ALIGNED(b, 64);
    // T* c = C.fArray;
    // ASSUME_ALIGNED(c, 64);

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      C(n, 0, 0) = A(n, 0, 0) * B(n, 0, 0) + A(n, 0, 1) * B(n, 1, 0);
      C(n, 0, 1) = A(n, 0, 0) * B(n, 0, 1) + A(n, 0, 1) * B(n, 1, 1);
      C(n, 0, 2) = A(n, 0, 0) * B(n, 0, 2) + A(n, 0, 1) * B(n, 1, 2);
      C(n, 0, 3) = 0;
      C(n, 0, 4) = 0;
      C(n, 0, 5) = 0;
      C(n, 0, 6) = A(n, 1, 0) * B(n, 0, 0) + A(n, 1, 1) * B(n, 1, 0);
      C(n, 0, 7) = A(n, 1, 0) * B(n, 0, 1) + A(n, 1, 1) * B(n, 1, 1);
      C(n, 0, 8) = A(n, 1, 0) * B(n, 0, 2) + A(n, 1, 1) * B(n, 1, 2);
      C(n, 0, 9) = 0;
      C(n, 0, 10) = 0;
      C(n, 0, 11) = 0;
      C(n, 0, 12) = A(n, 2, 0) * B(n, 0, 0) + A(n, 2, 1) * B(n, 1, 0);
      C(n, 0, 13) = A(n, 2, 0) * B(n, 0, 1) + A(n, 2, 1) * B(n, 1, 1);
      C(n, 0, 14) = A(n, 2, 0) * B(n, 0, 2) + A(n, 2, 1) * B(n, 1, 2);
      C(n, 0, 15) = 0;
      C(n, 0, 16) = 0;
      C(n, 0, 17) = 0;
      C(n, 0, 18) = A(n, 3, 0) * B(n, 0, 0) + A(n, 3, 1) * B(n, 1, 0);
      C(n, 0, 19) = A(n, 3, 0) * B(n, 0, 1) + A(n, 3, 1) * B(n, 1, 1);
      C(n, 0, 20) = A(n, 3, 0) * B(n, 0, 2) + A(n, 3, 1) * B(n, 1, 2);
      C(n, 0, 21) = 0;
      C(n, 0, 22) = 0;
      C(n, 0, 23) = 0;
      C(n, 0, 24) = A(n, 4, 0) * B(n, 0, 0) + A(n, 4, 1) * B(n, 1, 0);
      C(n, 0, 25) = A(n, 4, 0) * B(n, 0, 1) + A(n, 4, 1) * B(n, 1, 1);
      C(n, 0, 26) = A(n, 4, 0) * B(n, 0, 2) + A(n, 4, 1) * B(n, 1, 2);
      C(n, 0, 27) = 0;
      C(n, 0, 28) = 0;
      C(n, 0, 29) = 0;
      C(n, 0, 30) = A(n, 5, 0) * B(n, 0, 0) + A(n, 5, 1) * B(n, 1, 0);
      C(n, 0, 31) = A(n, 5, 0) * B(n, 0, 1) + A(n, 5, 1) * B(n, 1, 1);
      C(n, 0, 32) = A(n, 5, 0) * B(n, 0, 2) + A(n, 5, 1) * B(n, 1, 2);
      C(n, 0, 33) = 0;
      C(n, 0, 34) = 0;
      C(n, 0, 35) = 0;
    }
  }

  inline void KHC(const MPlexLL& A, const MPlexLS& B, MPlexLS& C) {
    // C = A * B, C is 6x6, A is 6x6 , B is 6x6 sym

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "KHC.ah"
  }

  inline void KHC(const MPlexL2& A, const MPlexLS& B, MPlexLS& C) {
    // C = A * B, C is 6x6 sym, A is 6x2 , B is 6x6 sym

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "K62HC.ah"
  }

  void JacCCS2Loc(const MPlex55& A, const MPlex56& B, MPlex56& C) {
    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "JacCCS2Loc.ah"
  }

  void PsErrLoc(const MPlex56& A, const MPlexLS& B, MPlex56& C) {
    // C = A * B

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "PsErrLoc.ah"
  }

  void PsErrLocTransp(const MPlex56& B, const MPlex56& A, MPlex5S& C) {
    // C = B * AT;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "PsErrLocTransp.ah"
  }

  void PsErrLocUpd(const MPlex55& A, const MPlex5S& B, MPlex5S& C) {
    // C = A * B;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "PsErrLocUpd.ah"
  }

  void JacLoc2CCS(const MPlex65& A, const MPlex55& B, MPlex65& C) {
    // C = A * B;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "JacLoc2CCS.ah"
  }

  void OutErrCCS(const MPlex65& A, const MPlex5S& B, MPlex65& C) {
    // C = A * B

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "OutErrCCS.ah"
  }

  void OutErrCCSTransp(const MPlex65& B, const MPlex65& A, MPlexLS& C) {
    // C = B * AT;

    typedef float T;
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "OutErrCCSTransp.ah"
  }

  //Warning: MultFull is not vectorized!
  template <typename T1, typename T2, typename T3>
  void MultFull(const T1& A, int nia, int nja, const T2& B, int nib, int njb, T3& C, int nic, int njc) {
#ifdef DEBUG
    assert(nja == nib);
    assert(nia == nic);
    assert(njb == njc);
#endif
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < nia; ++i) {
        for (int j = 0; j < njb; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < nja; ++k)
            C(n, i, j) += A.constAt(n, i, k) * B.constAt(n, k, j);
        }
      }
    }
  }

  //Warning: MultTranspFull is not vectorized!
  // (careful about which one is transposed, I think rows and cols are swapped and the one that is transposed is B)
  template <typename T1, typename T2, typename T3>
  void MultTranspFull(const T1& A, int nia, int nja, const T2& B, int nib, int njb, T3& C, int nic, int njc) {
#ifdef DEBUG
    assert(nja == njb);
    assert(nia == nic);
    assert(nib == njc);
#endif
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < nia; ++i) {
        for (int j = 0; j < nib; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < nja; ++k)
            C(n, i, j) += A.constAt(n, i, k) * B.constAt(n, j, k);
        }
      }
    }
  }

}  // namespace

//==============================================================================
// Kalman operations - common dummy variables
//==============================================================================

namespace {
  // Dummy variables for parameter consistency to kalmanOperation.
  // Through KalmanFilterOperation enum parameter it is guaranteed that
  // those will never get accessed in the code (read from or written into).

  CMS_SA_ALLOW mkfit::MPlexLS dummy_err;
  CMS_SA_ALLOW mkfit::MPlexLV dummy_par;
  CMS_SA_ALLOW mkfit::MPlexQF dummy_chi2;
}  // namespace

namespace mkfit {

  //==============================================================================
  // Kalman operations - Barrel
  //==============================================================================

  void kalmanUpdate(const MPlexLS& psErr,
                    const MPlexLV& psPar,
                    const MPlexHS& msErr,
                    const MPlexHV& msPar,
                    MPlexLS& outErr,
                    MPlexLV& outPar,
                    const int N_proc) {
    kalmanOperation(KFO_Update_Params | KFO_Local_Cov, psErr, psPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
  }

  void kalmanPropagateAndUpdate(const MPlexLS& psErr,
                                const MPlexLV& psPar,
                                MPlexQI& Chg,
                                const MPlexHS& msErr,
                                const MPlexHV& msPar,
                                MPlexLS& outErr,
                                MPlexLV& outPar,
                                MPlexQI& outFailFlag,
                                const int N_proc,
                                const PropagationFlags& propFlags,
                                const bool propToHit) {
    if (propToHit) {
      MPlexLS propErr;
      MPlexLV propPar;
      MPlexQF msRad;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msRad.At(n, 0, 0) = hipo(msPar.constAt(n, 0, 0), msPar.constAt(n, 1, 0));
      }

      propagateHelixToRMPlex(psErr, psPar, Chg, msRad, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperation(
          KFO_Update_Params | KFO_Local_Cov, propErr, propPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
    } else {
      kalmanOperation(
          KFO_Update_Params | KFO_Local_Cov, psErr, psPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
    }
    for (int n = 0; n < NN; ++n) {
      if (n < N_proc && outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  //------------------------------------------------------------------------------

  void kalmanComputeChi2(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& inChg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         MPlexQF& outChi2,
                         const int N_proc) {
    kalmanOperation(KFO_Calculate_Chi2, psErr, psPar, msErr, msPar, dummy_err, dummy_par, outChi2, N_proc);
  }

  void kalmanPropagateAndComputeChi2(const MPlexLS& psErr,
                                     const MPlexLV& psPar,
                                     const MPlexQI& inChg,
                                     const MPlexHS& msErr,
                                     const MPlexHV& msPar,
                                     MPlexQF& outChi2,
                                     MPlexLV& propPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags& propFlags,
                                     const bool propToHit) {
    propPar = psPar;
    if (propToHit) {
      MPlexLS propErr;
      MPlexQF msRad;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        if (n < N_proc) {
          msRad.At(n, 0, 0) = hipo(msPar.constAt(n, 0, 0), msPar.constAt(n, 1, 0));
        } else {
          msRad.At(n, 0, 0) = 0.0f;
        }
      }

      propagateHelixToRMPlex(psErr, psPar, inChg, msRad, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperation(KFO_Calculate_Chi2, propErr, propPar, msErr, msPar, dummy_err, dummy_par, outChi2, N_proc);
    } else {
      kalmanOperation(KFO_Calculate_Chi2, psErr, psPar, msErr, msPar, dummy_err, dummy_par, outChi2, N_proc);
    }
  }

  //------------------------------------------------------------------------------

  void kalmanOperation(const int kfOp,
                       const MPlexLS& psErr,
                       const MPlexLV& psPar,
                       const MPlexHS& msErr,
                       const MPlexHV& msPar,
                       MPlexLS& outErr,
                       MPlexLV& outPar,
                       MPlexQF& outChi2,
                       const int N_proc) {
#ifdef DEBUG
    {
      dmutex_guard;
      printf("psPar:\n");
      for (int i = 0; i < 6; ++i) {
        printf("%8f ", psPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("psErr:\n");
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
          printf("%8f ", psErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("msPar:\n");
      for (int i = 0; i < 3; ++i) {
        printf("%8f ", msPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("msErr:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", msErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    // Rotate global point on tangent plane to cylinder
    // Tangent point is half way between hit and propagate position

    // Rotation matrix
    //  rotT00  0  rotT01
    //  rotT01  0 -rotT00
    //     0    1    0
    // Minimize temporaries: only two float are needed!

    MPlexQF rotT00;
    MPlexQF rotT01;
    for (int n = 0; n < NN; ++n) {
      if (n < N_proc) {
        const float r = hipo(msPar.constAt(n, 0, 0), msPar.constAt(n, 1, 0));
        rotT00.At(n, 0, 0) = -(msPar.constAt(n, 1, 0) + psPar.constAt(n, 1, 0)) / (2 * r);
        rotT01.At(n, 0, 0) = (msPar.constAt(n, 0, 0) + psPar.constAt(n, 0, 0)) / (2 * r);
      } else {
        rotT00.At(n, 0, 0) = 0.0f;
        rotT01.At(n, 0, 0) = 0.0f;
      }
    }

    MPlexHV res_glo;  //position residual in global coordinates
    SubtractFirst3(msPar, psPar, res_glo);

    MPlexHS resErr_glo;  //covariance sum in global position coordinates
    AddIntoUpperLeft3x3(psErr, msErr, resErr_glo);

    MPlex2V res_loc;  //position residual in local coordinates
    RotateResidualsOnTangentPlane(rotT00, rotT01, res_glo, res_loc);
    MPlex2S resErr_loc;  //covariance sum in local position coordinates
    MPlexHH tempHH;
    ProjectResErr(rotT00, rotT01, resErr_glo, tempHH);
    ProjectResErrTransp(rotT00, rotT01, tempHH, resErr_loc);

#ifdef DEBUG
    {
      dmutex_guard;
      printf("res_glo:\n");
      for (int i = 0; i < 3; ++i) {
        printf("%8f ", res_glo.At(0, i, 0));
      }
      printf("\n");
      printf("resErr_glo:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", resErr_glo.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("res_loc:\n");
      for (int i = 0; i < 2; ++i) {
        printf("%8f ", res_loc.At(0, i, 0));
      }
      printf("\n");
      printf("tempHH:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", tempHH.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("resErr_loc:\n");
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
          printf("%8f ", resErr_loc.At(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    //invert the 2x2 matrix
    Matriplex::invertCramerSym(resErr_loc);

    if (kfOp & KFO_Calculate_Chi2) {
      Chi2Similarity(res_loc, resErr_loc, outChi2);

#ifdef DEBUG
      {
        dmutex_guard;
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("chi2: %8f\n", outChi2.At(0, 0, 0));
      }
#endif
    }

    if (kfOp & KFO_Update_Params) {
      MPlexLS psErrLoc = psErr;
      if (kfOp & KFO_Local_Cov)
        CovXYconstrain(rotT00, rotT01, psErr, psErrLoc);

      MPlexLH K;                                      // kalman gain, fixme should be L2
      KalmanHTG(rotT00, rotT01, resErr_loc, tempHH);  // intermediate term to get kalman gain (H^T*G)
      KalmanGain(psErrLoc, tempHH, K);

      MultResidualsAdd(K, psPar, res_loc, outPar);

      squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

      MPlexLL tempLL;
      KHMult(K, rotT00, rotT01, tempLL);
      KHC(tempLL, psErrLoc, outErr);
      outErr.subtract(psErrLoc, outErr);

#ifdef DEBUG
      {
        dmutex_guard;
        if (kfOp & KFO_Local_Cov) {
          printf("psErrLoc:\n");
          for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j)
              printf("% 8e ", psErrLoc.At(0, i, j));
            printf("\n");
          }
          printf("\n");
        }
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("tempHH:\n");
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j)
            printf("%8f ", tempHH.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("K:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 3; ++j)
            printf("%8f ", K.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("tempLL:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", tempLL.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("outPar:\n");
        for (int i = 0; i < 6; ++i) {
          printf("%8f  ", outPar.At(0, i, 0));
        }
        printf("\n");
        printf("outErr:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", outErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
      }
#endif
    }
  }

  //==============================================================================
  // Kalman operations - Plane
  //==============================================================================

  void kalmanUpdatePlane(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& Chg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         const MPlexHV& plNrm,
                         const MPlexHV& plDir,
                         const MPlexHV& plPnt,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         const int N_proc) {
    kalmanOperationPlaneLocal(KFO_Update_Params | KFO_Local_Cov,
                              psErr,
                              psPar,
                              Chg,
                              msErr,
                              msPar,
                              plNrm,
                              plDir,
                              plPnt,
                              outErr,
                              outPar,
                              dummy_chi2,
                              N_proc);
  }

  void kalmanPropagateAndUpdatePlane(const MPlexLS& psErr,
                                     const MPlexLV& psPar,
                                     MPlexQI& Chg,
                                     const MPlexHS& msErr,
                                     const MPlexHV& msPar,
                                     const MPlexHV& plNrm,
                                     const MPlexHV& plDir,
                                     const MPlexHV& plPnt,
                                     MPlexLS& outErr,
                                     MPlexLV& outPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags& propFlags,
                                     const bool propToHit) {
    if (propToHit) {
      MPlexLS propErr;
      MPlexLV propPar;
      propagateHelixToPlaneMPlex(psErr, psPar, Chg, plPnt, plNrm, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperationPlaneLocal(KFO_Update_Params | KFO_Local_Cov,
                                propErr,
                                propPar,
                                Chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                outErr,
                                outPar,
                                dummy_chi2,
                                N_proc);
    } else {
      kalmanOperationPlaneLocal(KFO_Update_Params | KFO_Local_Cov,
                                psErr,
                                psPar,
                                Chg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                outErr,
                                outPar,
                                dummy_chi2,
                                N_proc);
    }
    for (int n = 0; n < NN; ++n) {
      if (outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  //------------------------------------------------------------------------------

  void kalmanComputeChi2Plane(const MPlexLS& psErr,
                              const MPlexLV& psPar,
                              const MPlexQI& inChg,
                              const MPlexHS& msErr,
                              const MPlexHV& msPar,
                              const MPlexHV& plNrm,
                              const MPlexHV& plDir,
                              const MPlexHV& plPnt,
                              MPlexQF& outChi2,
                              const int N_proc) {
    kalmanOperationPlaneLocal(KFO_Calculate_Chi2,
                              psErr,
                              psPar,
                              inChg,
                              msErr,
                              msPar,
                              plNrm,
                              plDir,
                              plPnt,
                              dummy_err,
                              dummy_par,
                              outChi2,
                              N_proc);
  }

  void kalmanPropagateAndComputeChi2Plane(const MPlexLS& psErr,
                                          const MPlexLV& psPar,
                                          const MPlexQI& inChg,
                                          const MPlexHS& msErr,
                                          const MPlexHV& msPar,
                                          const MPlexHV& plNrm,
                                          const MPlexHV& plDir,
                                          const MPlexHV& plPnt,
                                          MPlexQF& outChi2,
                                          MPlexLV& propPar,
                                          MPlexQI& outFailFlag,
                                          const int N_proc,
                                          const PropagationFlags& propFlags,
                                          const bool propToHit) {
    propPar = psPar;
    if (propToHit) {
      MPlexLS propErr;
      propagateHelixToPlaneMPlex(psErr, psPar, inChg, plPnt, plNrm, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperationPlaneLocal(KFO_Calculate_Chi2,
                                propErr,
                                propPar,
                                inChg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                dummy_err,
                                dummy_par,
                                outChi2,
                                N_proc);
    } else {
      kalmanOperationPlaneLocal(KFO_Calculate_Chi2,
                                psErr,
                                psPar,
                                inChg,
                                msErr,
                                msPar,
                                plNrm,
                                plDir,
                                plPnt,
                                dummy_err,
                                dummy_par,
                                outChi2,
                                N_proc);
    }
  }

  //------------------------------------------------------------------------------

  void kalmanOperationPlaneLocal(const int kfOp,
                                 const MPlexLS& psErr,
                                 const MPlexLV& psPar,
                                 const MPlexQI& inChg,
                                 const MPlexHS& msErr,
                                 const MPlexHV& msPar,
                                 const MPlexHV& plNrm,
                                 const MPlexHV& plDir,
                                 const MPlexHV& plPnt,
                                 MPlexLS& outErr,
                                 MPlexLV& outPar,
                                 MPlexQF& outChi2,
                                 const int N_proc) {
#ifdef DEBUG
    {
      dmutex_guard;
      printf("psPar:\n");
      for (int i = 0; i < 6; ++i) {
        printf("%8f ", psPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("psErr:\n");
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
          printf("%8f ", psErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("msPar:\n");
      for (int i = 0; i < 3; ++i) {
        printf("%8f ", msPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("msErr:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", msErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    MPlexHH rot;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      rot(n, 0, 0) = plDir(n, 0, 0);
      rot(n, 0, 1) = plDir(n, 1, 0);
      rot(n, 0, 2) = plDir(n, 2, 0);
      rot(n, 1, 0) = plNrm(n, 1, 0) * plDir(n, 2, 0) - plNrm(n, 2, 0) * plDir(n, 1, 0);
      rot(n, 1, 1) = plNrm(n, 2, 0) * plDir(n, 0, 0) - plNrm(n, 0, 0) * plDir(n, 2, 0);
      rot(n, 1, 2) = plNrm(n, 0, 0) * plDir(n, 1, 0) - plNrm(n, 1, 0) * plDir(n, 0, 0);
      rot(n, 2, 0) = plNrm(n, 0, 0);
      rot(n, 2, 1) = plNrm(n, 1, 0);
      rot(n, 2, 2) = plNrm(n, 2, 0);
    }

    // get local parameters
    MPlexHV xd;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      xd(n, 0, 0) = psPar(n, 0, 0) - plPnt(n, 0, 0);
      xd(n, 0, 1) = psPar(n, 0, 1) - plPnt(n, 0, 1);
      xd(n, 0, 2) = psPar(n, 0, 2) - plPnt(n, 0, 2);
    }
    MPlex2V xlo;
    RotateResidualsOnPlane(rot, xd, xlo);

    MPlexQF sinP, sinT, cosP, cosT, pt;  //fixme VDT or something?
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pt(n, 0, 0) = 1.f / psPar(n, 3, 0);
      sinP(n, 0, 0) = std::sin(psPar(n, 4, 0));
      cosP(n, 0, 0) = std::cos(psPar(n, 4, 0));
      sinT(n, 0, 0) = std::sin(psPar(n, 5, 0));
      cosT(n, 0, 0) = std::cos(psPar(n, 5, 0));
    }

    MPlexHV pgl;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pgl(n, 0, 0) = cosP(n, 0, 0) * pt(n, 0, 0);
      pgl(n, 0, 1) = sinP(n, 0, 0) * pt(n, 0, 0);
      pgl(n, 0, 2) = cosT(n, 0, 0) * pt(n, 0, 0) / sinT(n, 0, 0);
    }

    MPlexHV plo;
    RotateVectorOnPlane(rot, pgl, plo);
    MPlex5V lp;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      lp(n, 0, 0) = inChg(n, 0, 0) * psPar(n, 3, 0) * sinT(n, 0, 0);
      lp(n, 0, 1) = plo(n, 0, 0) / plo(n, 0, 2);
      lp(n, 0, 2) = plo(n, 0, 1) / plo(n, 0, 2);
      lp(n, 0, 3) = xlo(n, 0, 0);
      lp(n, 0, 4) = xlo(n, 0, 1);
    }
    MPlexQI pzSign;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pzSign(n, 0, 0) = plo(n, 0, 2) > 0.f ? 1 : -1;
    }

    /*
    printf("rot:\n");
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j)
	printf("%8f ", rot.At(0, i, j));
      printf("\n");
    }
    printf("\n");
    printf("plPnt:\n");
    for (int i = 0; i < 3; ++i) {
      printf("%8f ", plPnt.constAt(0, 0, i));
    }
    printf("\n");
    printf("xlo:\n");
    for (int i = 0; i < 2; ++i) {
      printf("%8f ", xlo.At(0, i, 0));
    }
    printf("\n");
    printf("pgl:\n");
    for (int i = 0; i < 3; ++i) {
      printf("%8f ", pgl.At(0, i, 0));
    }
    printf("\n");
    printf("plo:\n");
    for (int i = 0; i < 3; ++i) {
      printf("%8f ", plo.At(0, i, 0));
    }
    printf("\n");
    printf("lp:\n");
    for (int i = 0; i < 5; ++i) {
      printf("%8f ", lp.At(0, i, 0));
    }
    printf("\n");
    */

    //now we need the jacobian to convert from CCS to curvilinear
    // code from TrackState::jacobianCCSToCurvilinear
    MPlex56 jacCCS2Curv(0.f);
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      jacCCS2Curv(n, 0, 3) = inChg(n, 0, 0) * sinT(n, 0, 0);
      jacCCS2Curv(n, 0, 5) = inChg(n, 0, 0) * cosT(n, 0, 0) * psPar(n, 3, 0);
      jacCCS2Curv(n, 1, 5) = -1.f;
      jacCCS2Curv(n, 2, 4) = 1.f;
      jacCCS2Curv(n, 3, 0) = -sinP(n, 0, 0);
      jacCCS2Curv(n, 3, 1) = cosP(n, 0, 0);
      jacCCS2Curv(n, 4, 0) = -cosP(n, 0, 0) * cosT(n, 0, 0);
      jacCCS2Curv(n, 4, 1) = -sinP(n, 0, 0) * cosT(n, 0, 0);
      jacCCS2Curv(n, 4, 2) = sinT(n, 0, 0);
    }

    //now we need the jacobian from curv to local
    // code from TrackingTools/AnalyticalJacobians/src/JacobianCurvilinearToLocal.cc
    MPlexHV un;
    MPlexHV vn;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      const float abslp00 = std::abs(lp(n, 0, 0));
      vn(n, 0, 2) = std::max(1.e-30f, abslp00 * pt(n, 0, 0));
      un(n, 0, 0) = -pgl(n, 0, 1) * abslp00 / vn(n, 0, 2);
      un(n, 0, 1) = pgl(n, 0, 0) * abslp00 / vn(n, 0, 2);
      un(n, 0, 2) = 0.f;
      vn(n, 0, 0) = -pgl(n, 0, 2) * abslp00 * un(n, 0, 1);
      vn(n, 0, 1) = pgl(n, 0, 2) * abslp00 * un(n, 0, 0);
    }
    MPlexHV u;
    RotateVectorOnPlane(rot, un, u);
    MPlexHV v;
    RotateVectorOnPlane(rot, vn, v);
    MPlex55 jacCurv2Loc(0.f);
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      // fixme? //(pf.use_param_b_field ? 0.01f * Const::sol * Config::bFieldFromZR(psPar(n, 2, 0), hipo(psPar(n, 0, 0), psPar(n, 1, 0))) : 0.01f * Const::sol * Config::Bfield);
      const float bF = 0.01f * Const::sol * Config::Bfield;
      const float qh2 = bF * lp(n, 0, 0);
      const float t1r = std::sqrt(1.f + lp(n, 0, 1) * lp(n, 0, 1) + lp(n, 0, 2) * lp(n, 0, 2)) * pzSign(n, 0, 0);
      const float t2r = t1r * t1r;
      const float t3r = t1r * t2r;
      jacCurv2Loc(n, 0, 0) = 1.f;
      jacCurv2Loc(n, 1, 1) = -u(n, 0, 1) * t2r;
      jacCurv2Loc(n, 1, 2) = v(n, 0, 1) * vn(n, 0, 2) * t2r;
      jacCurv2Loc(n, 2, 1) = u(n, 0, 0) * t2r;
      jacCurv2Loc(n, 2, 2) = -v(n, 0, 0) * vn(n, 0, 2) * t2r;
      jacCurv2Loc(n, 3, 3) = v(n, 0, 1) * t1r;
      jacCurv2Loc(n, 3, 4) = -u(n, 0, 1) * t1r;
      jacCurv2Loc(n, 4, 3) = -v(n, 0, 0) * t1r;
      jacCurv2Loc(n, 4, 4) = u(n, 0, 0) * t1r;
      const float cosz = -vn(n, 0, 2) * qh2;
      const float ui = u(n, 0, 2) * t3r;
      const float vi = v(n, 0, 2) * t3r;
      jacCurv2Loc(n, 1, 3) = -ui * v(n, 0, 1) * cosz;
      jacCurv2Loc(n, 1, 4) = -vi * v(n, 0, 1) * cosz;
      jacCurv2Loc(n, 2, 3) = ui * v(n, 0, 0) * cosz;
      jacCurv2Loc(n, 2, 4) = vi * v(n, 0, 0) * cosz;
      //
    }

    // jacobian for converting from CCS to Loc (via Curv)
    MPlex56 jacCCS2Loc;
    JacCCS2Loc(jacCurv2Loc, jacCCS2Curv, jacCCS2Loc);

    // local error!
    MPlex5S psErrLoc;
    MPlex56 temp56;
    PsErrLoc(jacCCS2Loc, psErr, temp56);
    PsErrLocTransp(temp56, jacCCS2Loc, psErrLoc);

    MPlexHV md;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      md(n, 0, 0) = msPar(n, 0, 0) - plPnt(n, 0, 0);
      md(n, 0, 1) = msPar(n, 0, 1) - plPnt(n, 0, 1);
      md(n, 0, 2) = msPar(n, 0, 2) - plPnt(n, 0, 2);
    }
    MPlex2V mslo;
    RotateResidualsOnPlane(rot, md, mslo);

    MPlex2V res_loc;  //position residual in local coordinates
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      res_loc(n, 0, 0) = mslo(n, 0, 0) - xlo(n, 0, 0);
      res_loc(n, 0, 1) = mslo(n, 0, 1) - xlo(n, 0, 1);
    }

    MPlex2S msErr_loc;
    MPlex2H temp2Hmsl;
    ProjectResErr(rot, msErr, temp2Hmsl);
    ProjectResErrTransp(rot, temp2Hmsl, msErr_loc);

    MPlex2S resErr_loc;  //covariance sum in local position coordinates
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      resErr_loc(n, 0, 0) = psErrLoc(n, 3, 3) + msErr_loc(n, 0, 0);
      resErr_loc(n, 0, 1) = psErrLoc(n, 3, 4) + msErr_loc(n, 0, 1);
      resErr_loc(n, 1, 1) = psErrLoc(n, 4, 4) + msErr_loc(n, 1, 1);
    }
    /*
    printf("jacCCS2Curv:\n");
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 6; ++j)
	printf("%8f ", jacCCS2Curv.At(0, i, j));
      printf("\n");
    }
    printf("un:\n");
    for (int i = 0; i < 3; ++i) {
      printf("%8f ", un.At(0, i, 0));
    }
    printf("\n");
    printf("u:\n");
    for (int i = 0; i < 3; ++i) {
      printf("%8f ", u.At(0, i, 0));
    }
    printf("\n");
    printf("\n");
    printf("jacCurv2Loc:\n");
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j)
	printf("%8f ", jacCurv2Loc.At(0, i, j));
      printf("\n");
    }
    printf("\n");
    printf("jacCCS2Loc:\n");
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 6; ++j)
	printf("%8f ", jacCCS2Loc.At(0, i, j));
      printf("\n");
    }
    printf("\n");
    printf("temp56:\n");
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 6; ++j)
	printf("%8f ", temp56.At(0, i, j));
      printf("\n");
    }
    printf("\n");
    printf("psErrLoc:\n");
    for (int i = 0; i < 5; ++i) {
      for (int j = 0; j < 5; ++j)
	printf("%8f ", psErrLoc.At(0, i, j));
      printf("\n");
    }
    printf("\n");
    printf("res_loc:\n");
    for (int i = 0; i < 2; ++i) {
      printf("%8f ", res_loc.At(0, i, 0));
    }
    printf("\n");
    printf("resErr_loc:\n");
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j)
	printf("%8f ", resErr_loc.At(0, i, j));
      printf("\n");
    }
    printf("\n");
    */
    //invert the 2x2 matrix
    Matriplex::invertCramerSym(resErr_loc);

    if (kfOp & KFO_Calculate_Chi2) {
      Chi2Similarity(res_loc, resErr_loc, outChi2);

#ifdef DEBUG
      {
        dmutex_guard;
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("chi2: %8f\n", outChi2.At(0, 0, 0));
      }
#endif
    }

    if (kfOp & KFO_Update_Params) {
      MPlex52 K;  // kalman gain
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
#pragma GCC unroll 5
        for (int j = 0; j < 5; ++j) {
          K(n, j, 0) = resErr_loc(n, 0, 0) * psErrLoc(n, j, 3) + resErr_loc(n, 0, 1) * psErrLoc(n, j, 4);
          K(n, j, 1) = resErr_loc(n, 0, 1) * psErrLoc(n, j, 3) + resErr_loc(n, 1, 1) * psErrLoc(n, j, 4);
        }
      }

      MPlex5V lp_upd;
      MultResidualsAdd(K, lp, res_loc, lp_upd);

      MPlex55 ImKH(0.f);
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
#pragma GCC unroll 5
        for (int j = 0; j < 5; ++j) {
          ImKH(n, j, j) = 1.f;
          ImKH(n, j, 3) -= K(n, j, 0);
          ImKH(n, j, 4) -= K(n, j, 1);
        }
      }
      MPlex5S psErrLoc_upd;
      PsErrLocUpd(ImKH, psErrLoc, psErrLoc_upd);

      //convert local updated parameters into CCS
      MPlexHV lxu;
      MPlexHV lpu;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        lxu(n, 0, 0) = lp_upd(n, 0, 3);
        lxu(n, 0, 1) = lp_upd(n, 0, 4);
        lxu(n, 0, 2) = 0.f;
        lpu(n, 0, 2) =
            pzSign(n, 0, 0) / (std::max(std::abs(lp_upd(n, 0, 0)), 1.e-9f) *
                               std::sqrt(1.f + lp_upd(n, 0, 1) * lp_upd(n, 0, 1) + lp_upd(n, 0, 2) * lp_upd(n, 0, 2)));
        lpu(n, 0, 0) = lpu(n, 0, 2) * lp_upd(n, 0, 1);
        lpu(n, 0, 1) = lpu(n, 0, 2) * lp_upd(n, 0, 2);
      }
      MPlexHV gxu;
      RotateVectorOnPlaneTransp(rot, lxu, gxu);
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        gxu(n, 0, 0) += plPnt(n, 0, 0);
        gxu(n, 0, 1) += plPnt(n, 0, 1);
        gxu(n, 0, 2) += plPnt(n, 0, 2);
      }
      MPlexHV gpu;
      RotateVectorOnPlaneTransp(rot, lpu, gpu);

      MPlexQF p;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        pt(n, 0, 0) = std::sqrt(gpu.At(n, 0, 0) * gpu.At(n, 0, 0) + gpu.At(n, 0, 1) * gpu.At(n, 0, 1));
        p(n, 0, 0) = std::sqrt(pt.At(n, 0, 0) * pt.At(n, 0, 0) + gpu.At(n, 0, 2) * gpu.At(n, 0, 2));
        sinP(n, 0, 0) = gpu.At(n, 0, 1) / pt(n, 0, 0);
        cosP(n, 0, 0) = gpu.At(n, 0, 0) / pt(n, 0, 0);
        sinT(n, 0, 0) = pt(n, 0, 0) / p(n, 0, 0);
        cosT(n, 0, 0) = gpu.At(n, 0, 2) / p(n, 0, 0);
      }

#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        outPar(n, 0, 0) = gxu.At(n, 0, 0);
        outPar(n, 0, 1) = gxu.At(n, 0, 1);
        outPar(n, 0, 2) = gxu.At(n, 0, 2);
        outPar(n, 0, 3) = 1.f / pt(n, 0, 0);
        outPar(n, 0, 4) = getPhi(gpu.At(n, 0, 0), gpu.At(n, 0, 1));  //fixme VDT or something?
        outPar(n, 0, 5) = getTheta(pt(n, 0, 0), gpu.At(n, 0, 2));
      }

      //now we need the jacobian to convert from curvilinear to CCS
      // code from TrackState::jacobianCurvilinearToCCS
      MPlex65 jacCurv2CCS(0.f);
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        jacCurv2CCS(n, 0, 3) = -sinP(n, 0, 0);
        jacCurv2CCS(n, 0, 4) = -cosT(n, 0, 0) * cosP(n, 0, 0);
        jacCurv2CCS(n, 1, 3) = cosP(n, 0, 0);
        jacCurv2CCS(n, 1, 4) = -cosT(n, 0, 0) * sinP(n, 0, 0);
        jacCurv2CCS(n, 2, 4) = sinT(n, 0, 0);
        jacCurv2CCS(n, 3, 0) = inChg(n, 0, 0) / sinT(n, 0, 0);
        jacCurv2CCS(n, 3, 1) = outPar(n, 3, 0) * cosT(n, 0, 0) / sinT(n, 0, 0);
        jacCurv2CCS(n, 4, 2) = 1.f;
        jacCurv2CCS(n, 5, 1) = -1.f;
      }

      //now we need the jacobian from local to curv
      // code from TrackingTools/AnalyticalJacobians/src/JacobianLocalToCurvilinear.cc
      MPlexHV tnl;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        const float abslpupd00 = std::max(std::abs(lp_upd(n, 0, 0)), 1.e-9f);
        tnl(n, 0, 0) = lpu(n, 0, 0) * abslpupd00;
        tnl(n, 0, 1) = lpu(n, 0, 1) * abslpupd00;
        tnl(n, 0, 2) = lpu(n, 0, 2) * abslpupd00;
      }
      MPlexHV tn;
      RotateVectorOnPlaneTransp(rot, tnl, tn);
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        vn(n, 0, 2) = std::max(1.e-30f, std::sqrt(tn(n, 0, 0) * tn(n, 0, 0) + tn(n, 0, 1) * tn(n, 0, 1)));
        un(n, 0, 0) = -tn(n, 0, 1) / vn(n, 0, 2);
        un(n, 0, 1) = tn(n, 0, 0) / vn(n, 0, 2);
        un(n, 0, 2) = 0.f;
        vn(n, 0, 0) = -tn(n, 0, 2) * un(n, 0, 1);
        vn(n, 0, 1) = tn(n, 0, 2) * un(n, 0, 0);
      }
      MPlex55 jacLoc2Curv(0.f);
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        // fixme? //(pf.use_param_b_field ? 0.01f * Const::sol * Config::bFieldFromZR(psPar(n, 2, 0), hipo(psPar(n, 0, 0), psPar(n, 1, 0))) : 0.01f * Const::sol * Config::Bfield);
        const float bF = 0.01f * Const::sol * Config::Bfield;  //fixme: cache?
        const float qh2 = bF * lp_upd(n, 0, 0);
        const float cosl1 = 1.f / vn(n, 0, 2);
        const float uj = un(n, 0, 0) * rot(n, 0, 0) + un(n, 0, 1) * rot(n, 0, 1);
        const float uk = un(n, 0, 0) * rot(n, 1, 0) + un(n, 0, 1) * rot(n, 1, 1);
        const float vj = vn(n, 0, 0) * rot(n, 0, 0) + vn(n, 0, 1) * rot(n, 0, 1) + vn(n, 0, 2) * rot(n, 0, 2);
        const float vk = vn(n, 0, 0) * rot(n, 1, 0) + vn(n, 0, 1) * rot(n, 1, 1) + vn(n, 0, 2) * rot(n, 1, 2);
        const float cosz = vn(n, 0, 2) * qh2;
        jacLoc2Curv(n, 0, 0) = 1.f;
        jacLoc2Curv(n, 1, 1) = tnl(n, 0, 2) * vj;
        jacLoc2Curv(n, 1, 2) = tnl(n, 0, 2) * vk;
        jacLoc2Curv(n, 2, 1) = tnl(n, 0, 2) * uj * cosl1;
        jacLoc2Curv(n, 2, 2) = tnl(n, 0, 2) * uk * cosl1;
        jacLoc2Curv(n, 3, 3) = uj;
        jacLoc2Curv(n, 3, 4) = uk;
        jacLoc2Curv(n, 4, 3) = vj;
        jacLoc2Curv(n, 4, 4) = vk;
        jacLoc2Curv(n, 2, 3) = tnl(n, 0, 0) * (cosz * cosl1);
        jacLoc2Curv(n, 2, 4) = tnl(n, 0, 1) * (cosz * cosl1);
      }

      // jacobian for converting from Loc to CCS (via Curv)
      MPlex65 jacLoc2CCS;
      JacLoc2CCS(jacCurv2CCS, jacLoc2Curv, jacLoc2CCS);

      // CCS error!
      MPlex65 temp65;
      OutErrCCS(jacLoc2CCS, psErrLoc_upd, temp65);
      OutErrCCSTransp(temp65, jacLoc2CCS, outErr);

      /*
      printf("\n");
      printf("lp_upd:\n");
      for (int i = 0; i < 5; ++i) {
	printf("%8f ", lp_upd.At(0, i, 0));
      }
      printf("\n");
      printf("psErrLoc_upd:\n");
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j)
          printf("%8f ", psErrLoc_upd.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("lxu:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", lxu.At(0, i, 0));
      }
      printf("\n");
      printf("lpu:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", lpu.At(0, i, 0));
      }
      printf("\n");
      printf("gxu:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", gxu.At(0, i, 0));
      }
      printf("\n");
      printf("gpu:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", gpu.At(0, i, 0));
      }
      printf("\n");
      printf("outPar:\n");
      for (int i = 0; i < 6; ++i) {
	printf("%8f ", outPar.At(0, i, 0));
      }
      printf("\n");
      printf("tnl:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", tnl.At(0, i, 0));
      }
      printf("\n");
      printf("tn:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", tn.At(0, i, 0));
      }
      printf("\n");
      printf("un:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", un.At(0, i, 0));
      }
      printf("\n");
      printf("vn:\n");
      for (int i = 0; i < 3; ++i) {
	printf("%8f ", vn.At(0, i, 0));
      }
      printf("\n");
      printf("jacLoc2Curv:\n");
      for (int i = 0; i < 5; ++i) {
	for (int j = 0; j < 5; ++j)
	  printf("%8f ", jacLoc2Curv.At(0, i, j));
	printf("\n");
      }
      printf("\n");
      printf("outErr:\n");
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
          printf("%8f ", outErr.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      */

#ifdef DEBUG
      {
        dmutex_guard;
        if (kfOp & KFO_Local_Cov) {
          printf("psErrLoc_upd:\n");
          for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j)
              printf("% 8e ", psErrLoc_upd.At(0, i, j));
            printf("\n");
          }
          printf("\n");
        }
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("K:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", K.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("outPar:\n");
        for (int i = 0; i < 6; ++i) {
          printf("%8f  ", outPar.At(0, i, 0));
        }
        printf("\n");
        printf("outErr:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", outErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
      }
#endif
    }

    return;
  }

  //------------------------------------------------------------------------------

  void kalmanOperationPlane(const int kfOp,
                            const MPlexLS& psErr,
                            const MPlexLV& psPar,
                            const MPlexQI& inChg,
                            const MPlexHS& msErr,
                            const MPlexHV& msPar,
                            const MPlexHV& plNrm,
                            const MPlexHV& plDir,
                            const MPlexHV& plPnt,  //not used, can be removed (fixme)
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            MPlexQF& outChi2,
                            const int N_proc) {
#ifdef DEBUG
    {
      dmutex_guard;
      printf("psPar:\n");
      for (int i = 0; i < 6; ++i) {
        printf("%8f ", psPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("psErr:\n");
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
          printf("%8f ", psErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("msPar:\n");
      for (int i = 0; i < 3; ++i) {
        printf("%8f ", msPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("msErr:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", msErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    // Rotate global point on local plane

    // Rotation matrix
    //    D0  D1   D2
    //    X0  X1   X2
    //    N0  N1   N2
    // where D is the strip direction vector plDir, N is the normal plNrm, and X is the cross product between the two

    MPlex2H prj;
    for (int n = 0; n < NN; ++n) {
      prj(n, 0, 0) = plDir(n, 0, 0);
      prj(n, 0, 1) = plDir(n, 1, 0);
      prj(n, 0, 2) = plDir(n, 2, 0);
      prj(n, 1, 0) = plNrm(n, 1, 0) * plDir(n, 2, 0) - plNrm(n, 2, 0) * plDir(n, 1, 0);
      prj(n, 1, 1) = plNrm(n, 2, 0) * plDir(n, 0, 0) - plNrm(n, 0, 0) * plDir(n, 2, 0);
      prj(n, 1, 2) = plNrm(n, 0, 0) * plDir(n, 1, 0) - plNrm(n, 1, 0) * plDir(n, 0, 0);
    }

    MPlexHV res_glo;  //position residual in global coordinates
    SubtractFirst3(msPar, psPar, res_glo);

    MPlexHS resErr_glo;  //covariance sum in global position coordinates
    AddIntoUpperLeft3x3(psErr, msErr, resErr_glo);

    MPlex2V res_loc;  //position residual in local coordinates
    RotateResidualsOnPlane(prj, res_glo, res_loc);
    MPlex2S resErr_loc;  //covariance sum in local position coordinates
    MPlex2H temp2H;
    ProjectResErr(prj, resErr_glo, temp2H);
    ProjectResErrTransp(prj, temp2H, resErr_loc);

#ifdef DEBUG
    {
      dmutex_guard;
      printf("prj:\n");
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", prj.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("res_glo:\n");
      for (int i = 0; i < 3; ++i) {
        printf("%8f ", res_glo.At(0, i, 0));
      }
      printf("\n");
      printf("resErr_glo:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", resErr_glo.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("res_loc:\n");
      for (int i = 0; i < 2; ++i) {
        printf("%8f ", res_loc.At(0, i, 0));
      }
      printf("\n");
      printf("temp2H:\n");
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", temp2H.At(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("resErr_loc:\n");
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
          printf("%8f ", resErr_loc.At(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    //invert the 2x2 matrix
    Matriplex::invertCramerSym(resErr_loc);

    if (kfOp & KFO_Calculate_Chi2) {
      Chi2Similarity(res_loc, resErr_loc, outChi2);

#ifdef DEBUG
      {
        dmutex_guard;
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("chi2: %8f\n", outChi2.At(0, 0, 0));
      }
#endif
    }

    if (kfOp & KFO_Update_Params) {
      MPlexLS psErrLoc = psErr;

      MPlexH2 tempH2;
      MPlexL2 K;                           // kalman gain
      KalmanHTG(prj, resErr_loc, tempH2);  // intermediate term to get kalman gain (H^T*G)
      KalmanGain(psErrLoc, tempH2, K);

      MultResidualsAdd(K, psPar, res_loc, outPar);

      squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

      MPlexLL tempLL;
      KHMult(K, prj, tempLL);
      KHC(tempLL, psErrLoc, outErr);
      outErr.subtract(psErrLoc, outErr);

#ifdef DEBUG
      {
        dmutex_guard;
        if (kfOp & KFO_Local_Cov) {
          printf("psErrLoc:\n");
          for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j)
              printf("% 8e ", psErrLoc.At(0, i, j));
            printf("\n");
          }
          printf("\n");
        }
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("tempH2:\n");
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", tempH2.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("K:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", K.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("tempLL:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", tempLL.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("outPar:\n");
        for (int i = 0; i < 6; ++i) {
          printf("%8f  ", outPar.At(0, i, 0));
        }
        printf("\n");
        printf("outErr:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", outErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
      }
#endif
    }
  }

  //==============================================================================
  // Kalman operations - Endcap
  //==============================================================================

  void kalmanUpdateEndcap(const MPlexLS& psErr,
                          const MPlexLV& psPar,
                          const MPlexHS& msErr,
                          const MPlexHV& msPar,
                          MPlexLS& outErr,
                          MPlexLV& outPar,
                          const int N_proc) {
    kalmanOperationEndcap(KFO_Update_Params, psErr, psPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
  }

  void kalmanPropagateAndUpdateEndcap(const MPlexLS& psErr,
                                      const MPlexLV& psPar,
                                      MPlexQI& Chg,
                                      const MPlexHS& msErr,
                                      const MPlexHV& msPar,
                                      MPlexLS& outErr,
                                      MPlexLV& outPar,
                                      MPlexQI& outFailFlag,
                                      const int N_proc,
                                      const PropagationFlags& propFlags,
                                      const bool propToHit) {
    if (propToHit) {
      MPlexLS propErr;
      MPlexLV propPar;
      MPlexQF msZ;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msZ.At(n, 0, 0) = msPar.constAt(n, 2, 0);
      }

      propagateHelixToZMPlex(psErr, psPar, Chg, msZ, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperationEndcap(KFO_Update_Params, propErr, propPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
    } else {
      kalmanOperationEndcap(KFO_Update_Params, psErr, psPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
    }
    for (int n = 0; n < NN; ++n) {
      if (n < N_proc && outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  //------------------------------------------------------------------------------

  void kalmanComputeChi2Endcap(const MPlexLS& psErr,
                               const MPlexLV& psPar,
                               const MPlexQI& inChg,
                               const MPlexHS& msErr,
                               const MPlexHV& msPar,
                               MPlexQF& outChi2,
                               const int N_proc) {
    kalmanOperationEndcap(KFO_Calculate_Chi2, psErr, psPar, msErr, msPar, dummy_err, dummy_par, outChi2, N_proc);
  }

  void kalmanPropagateAndComputeChi2Endcap(const MPlexLS& psErr,
                                           const MPlexLV& psPar,
                                           const MPlexQI& inChg,
                                           const MPlexHS& msErr,
                                           const MPlexHV& msPar,
                                           MPlexQF& outChi2,
                                           MPlexLV& propPar,
                                           MPlexQI& outFailFlag,
                                           const int N_proc,
                                           const PropagationFlags& propFlags,
                                           const bool propToHit) {
    propPar = psPar;
    if (propToHit) {
      MPlexLS propErr;
      MPlexQF msZ;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msZ.At(n, 0, 0) = msPar.constAt(n, 2, 0);
      }

      propagateHelixToZMPlex(psErr, psPar, inChg, msZ, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperationEndcap(KFO_Calculate_Chi2, propErr, propPar, msErr, msPar, dummy_err, dummy_par, outChi2, N_proc);
    } else {
      kalmanOperationEndcap(KFO_Calculate_Chi2, psErr, psPar, msErr, msPar, dummy_err, dummy_par, outChi2, N_proc);
    }
  }

  //------------------------------------------------------------------------------

  void kalmanOperationEndcap(const int kfOp,
                             const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             MPlexQF& outChi2,
                             const int N_proc) {
#ifdef DEBUG
    {
      dmutex_guard;
      printf("updateParametersEndcapMPlex\n");
      printf("psPar:\n");
      for (int i = 0; i < 6; ++i) {
        printf("%8f ", psPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("msPar:\n");
      for (int i = 0; i < 3; ++i) {
        printf("%8f ", msPar.constAt(0, 0, i));
        printf("\n");
      }
      printf("\n");
      printf("psErr:\n");
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
          printf("%8f ", psErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
      printf("msErr:\n");
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
          printf("%8f ", msErr.constAt(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    MPlex2V res;
    SubtractFirst2(msPar, psPar, res);

    MPlex2S resErr;
    AddIntoUpperLeft2x2(psErr, msErr, resErr);

#ifdef DEBUG
    {
      dmutex_guard;
      printf("resErr:\n");
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
          printf("%8f ", resErr.At(0, i, j));
        printf("\n");
      }
      printf("\n");
    }
#endif

    //invert the 2x2 matrix
    Matriplex::invertCramerSym(resErr);

    if (kfOp & KFO_Calculate_Chi2) {
      Chi2Similarity(res, resErr, outChi2);

#ifdef DEBUG
      {
        dmutex_guard;
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("chi2: %8f\n", outChi2.At(0, 0, 0));
      }
#endif
    }

    if (kfOp & KFO_Update_Params) {
      MPlexL2 K;
      KalmanGain(psErr, resErr, K);

      MultResidualsAdd(K, psPar, res, outPar);

      squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

      KHC(K, psErr, outErr);

#ifdef DEBUG
      {
        dmutex_guard;
        printf("outErr before subtract:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", outErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
      }
#endif

      outErr.subtract(psErr, outErr);

#ifdef DEBUG
      {
        dmutex_guard;
        printf("res:\n");
        for (int i = 0; i < 2; ++i) {
          printf("%8f ", res.At(0, i, 0));
        }
        printf("\n");
        printf("resErr (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("K:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", K.At(0, i, j));
          printf("\n");
        }
        printf("\n");
        printf("outPar:\n");
        for (int i = 0; i < 6; ++i) {
          printf("%8f  ", outPar.At(0, i, 0));
        }
        printf("\n");
        printf("outErr:\n");
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            printf("%8f ", outErr.At(0, i, j));
          printf("\n");
        }
        printf("\n");
      }
#endif
    }
  }

}  // end namespace mkfit
