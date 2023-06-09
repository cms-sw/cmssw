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
    // where right half of kalman gain is 0

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
  // (careful about which one is transposed, I think rows and cols are swapped and the one that is transposed is A)
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
        msRad.At(n, 0, 0) = std::hypot(msPar.constAt(n, 0, 0), msPar.constAt(n, 1, 0));
      }

      propagateHelixToRMPlex(psErr, psPar, Chg, msRad, propErr, propPar, outFailFlag, N_proc, propFlags);

      kalmanOperation(
          KFO_Update_Params | KFO_Local_Cov, propErr, propPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
    } else {
      kalmanOperation(
          KFO_Update_Params | KFO_Local_Cov, psErr, psPar, msErr, msPar, outErr, outPar, dummy_chi2, N_proc);
    }
    for (int n = 0; n < NN; ++n) {
      if (outPar.At(n, 3, 0) < 0) {
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
        msRad.At(n, 0, 0) = std::hypot(msPar.constAt(n, 0, 0), msPar.constAt(n, 1, 0));
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
      const float r = std::hypot(msPar.constAt(n, 0, 0), msPar.constAt(n, 1, 0));
      rotT00.At(n, 0, 0) = -(msPar.constAt(n, 1, 0) + psPar.constAt(n, 1, 0)) / (2 * r);
      rotT01.At(n, 0, 0) = (msPar.constAt(n, 0, 0) + psPar.constAt(n, 0, 0)) / (2 * r);
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
      MPlexLL tempLL;

      squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

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
        printf("res_glo:\n");
        for (int i = 0; i < 3; ++i) {
          printf("%8f ", res_glo.At(0, i, 0));
        }
        printf("\n");
        printf("res_loc:\n");
        for (int i = 0; i < 2; ++i) {
          printf("%8f ", res_loc.At(0, i, 0));
        }
        printf("\n");
        printf("resErr_loc (Inv):\n");
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j)
            printf("%8f ", resErr_loc.At(0, i, j));
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
      if (outPar.At(n, 3, 0) < 0) {
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
