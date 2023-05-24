#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

//==============================================================================
// propagateLineToRMPlex
//==============================================================================

using namespace Matriplex;

namespace mkfit {

  void propagateLineToRMPlex(const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             const int N_proc) {
    // XXX Regenerate parts below with a script.

    const idx_t N = NN;

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      const float cosA = (psPar[0 * N + n] * psPar[3 * N + n] + psPar[1 * N + n] * psPar[4 * N + n]) /
                         (std::sqrt((psPar[0 * N + n] * psPar[0 * N + n] + psPar[1 * N + n] * psPar[1 * N + n]) *
                                    (psPar[3 * N + n] * psPar[3 * N + n] + psPar[4 * N + n] * psPar[4 * N + n])));
      const float dr = (hipo(msPar[0 * N + n], msPar[1 * N + n]) - hipo(psPar[0 * N + n], psPar[1 * N + n])) / cosA;

      dprint_np(n, "propagateLineToRMPlex dr=" << dr);

      const float pt = hipo(psPar[3 * N + n], psPar[4 * N + n]);
      const float p = dr / pt;  // path
      const float psq = p * p;

      outPar[0 * N + n] = psPar[0 * N + n] + p * psPar[3 * N + n];
      outPar[1 * N + n] = psPar[1 * N + n] + p * psPar[4 * N + n];
      outPar[2 * N + n] = psPar[2 * N + n] + p * psPar[5 * N + n];
      outPar[3 * N + n] = psPar[3 * N + n];
      outPar[4 * N + n] = psPar[4 * N + n];
      outPar[5 * N + n] = psPar[5 * N + n];

      {
        const MPlexLS& A = psErr;
        MPlexLS& B = outErr;

        B.fArray[0 * N + n] = A.fArray[0 * N + n];
        B.fArray[1 * N + n] = A.fArray[1 * N + n];
        B.fArray[2 * N + n] = A.fArray[2 * N + n];
        B.fArray[3 * N + n] = A.fArray[3 * N + n];
        B.fArray[4 * N + n] = A.fArray[4 * N + n];
        B.fArray[5 * N + n] = A.fArray[5 * N + n];
        B.fArray[6 * N + n] = A.fArray[6 * N + n] + p * A.fArray[0 * N + n];
        B.fArray[7 * N + n] = A.fArray[7 * N + n] + p * A.fArray[1 * N + n];
        B.fArray[8 * N + n] = A.fArray[8 * N + n] + p * A.fArray[3 * N + n];
        B.fArray[9 * N + n] =
            A.fArray[9 * N + n] + p * (A.fArray[6 * N + n] + A.fArray[6 * N + n]) + psq * A.fArray[0 * N + n];
        B.fArray[10 * N + n] = A.fArray[10 * N + n] + p * A.fArray[1 * N + n];
        B.fArray[11 * N + n] = A.fArray[11 * N + n] + p * A.fArray[2 * N + n];
        B.fArray[12 * N + n] = A.fArray[12 * N + n] + p * A.fArray[4 * N + n];
        B.fArray[13 * N + n] =
            A.fArray[13 * N + n] + p * (A.fArray[7 * N + n] + A.fArray[10 * N + n]) + psq * A.fArray[1 * N + n];
        B.fArray[14 * N + n] =
            A.fArray[14 * N + n] + p * (A.fArray[11 * N + n] + A.fArray[11 * N + n]) + psq * A.fArray[2 * N + n];
        B.fArray[15 * N + n] = A.fArray[15 * N + n] + p * A.fArray[3 * N + n];
        B.fArray[16 * N + n] = A.fArray[16 * N + n] + p * A.fArray[4 * N + n];
        B.fArray[17 * N + n] = A.fArray[17 * N + n] + p * A.fArray[5 * N + n];
        B.fArray[18 * N + n] =
            A.fArray[18 * N + n] + p * (A.fArray[8 * N + n] + A.fArray[15 * N + n]) + psq * A.fArray[3 * N + n];
        B.fArray[19 * N + n] =
            A.fArray[19 * N + n] + p * (A.fArray[12 * N + n] + A.fArray[16 * N + n]) + psq * A.fArray[4 * N + n];
        B.fArray[20 * N + n] =
            A.fArray[20 * N + n] + p * (A.fArray[17 * N + n] + A.fArray[17 * N + n]) + psq * A.fArray[5 * N + n];
      }

      dprint_np(n, "propagateLineToRMPlex arrive at r=" << hipo(outPar[0 * N + n], outPar[1 * N + n]));
    }
  }

}  // end namespace mkfit

//==============================================================================
// propagateHelixToRMPlex
//==============================================================================

namespace {
  using namespace mkfit;

  void MultHelixProp(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    // C = A * B

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixProp.ah"
  }

  void MultHelixPropTransp(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
    // C = B * AT;

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPropTransp.ah"
  }

  void MultHelixPropEndcap(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    // C = A * B

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPropEndcap.ah"
  }

  void MultHelixPropTranspEndcap(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
    // C = B * AT;

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPropTranspEndcap.ah"
  }

  inline void MultHelixPropTemp(const MPlexLL& A, const MPlexLL& B, MPlexLL& C, int n) {
    // C = A * B

    typedef float T;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

    c[0 * N + n] = a[0 * N + n] * b[0 * N + n] + a[1 * N + n] * b[6 * N + n] + a[2 * N + n] * b[12 * N + n] +
                   a[4 * N + n] * b[24 * N + n];
    c[1 * N + n] = a[0 * N + n] * b[1 * N + n] + a[1 * N + n] * b[7 * N + n] + a[2 * N + n] * b[13 * N + n] +
                   a[4 * N + n] * b[25 * N + n];
    c[2 * N + n] = a[2 * N + n];
    c[3 * N + n] = a[0 * N + n] * b[3 * N + n] + a[1 * N + n] * b[9 * N + n] + a[2 * N + n] * b[15 * N + n] +
                   a[3 * N + n] + a[4 * N + n] * b[27 * N + n];
    c[4 * N + n] = a[0 * N + n] * b[4 * N + n] + a[1 * N + n] * b[10 * N + n] + a[4 * N + n];
    c[5 * N + n] = a[2 * N + n] * b[17 * N + n] + a[5 * N + n];
    c[6 * N + n] = a[6 * N + n] * b[0 * N + n] + a[7 * N + n] * b[6 * N + n] + a[8 * N + n] * b[12 * N + n] +
                   a[10 * N + n] * b[24 * N + n];
    c[7 * N + n] = a[6 * N + n] * b[1 * N + n] + a[7 * N + n] * b[7 * N + n] + a[8 * N + n] * b[13 * N + n] +
                   a[10 * N + n] * b[25 * N + n];
    c[8 * N + n] = a[8 * N + n];
    c[9 * N + n] = a[6 * N + n] * b[3 * N + n] + a[7 * N + n] * b[9 * N + n] + a[8 * N + n] * b[15 * N + n] +
                   a[9 * N + n] + a[10 * N + n] * b[27 * N + n];
    c[10 * N + n] = a[6 * N + n] * b[4 * N + n] + a[7 * N + n] * b[10 * N + n] + a[10 * N + n];
    c[11 * N + n] = a[8 * N + n] * b[17 * N + n] + a[11 * N + n];
    c[12 * N + n] = a[12 * N + n] * b[0 * N + n] + a[13 * N + n] * b[6 * N + n] + a[14 * N + n] * b[12 * N + n] +
                    a[16 * N + n] * b[24 * N + n];
    c[13 * N + n] = a[12 * N + n] * b[1 * N + n] + a[13 * N + n] * b[7 * N + n] + a[14 * N + n] * b[13 * N + n] +
                    a[16 * N + n] * b[25 * N + n];
    c[14 * N + n] = a[14 * N + n];
    c[15 * N + n] = a[12 * N + n] * b[3 * N + n] + a[13 * N + n] * b[9 * N + n] + a[14 * N + n] * b[15 * N + n] +
                    a[15 * N + n] + a[16 * N + n] * b[27 * N + n];
    c[16 * N + n] = a[12 * N + n] * b[4 * N + n] + a[13 * N + n] * b[10 * N + n] + a[16 * N + n];
    c[17 * N + n] = a[14 * N + n] * b[17 * N + n] + a[17 * N + n];
    c[18 * N + n] = a[18 * N + n] * b[0 * N + n] + a[19 * N + n] * b[6 * N + n] + a[20 * N + n] * b[12 * N + n] +
                    a[22 * N + n] * b[24 * N + n];
    c[19 * N + n] = a[18 * N + n] * b[1 * N + n] + a[19 * N + n] * b[7 * N + n] + a[20 * N + n] * b[13 * N + n] +
                    a[22 * N + n] * b[25 * N + n];
    c[20 * N + n] = a[20 * N + n];
    c[21 * N + n] = a[18 * N + n] * b[3 * N + n] + a[19 * N + n] * b[9 * N + n] + a[20 * N + n] * b[15 * N + n] +
                    a[21 * N + n] + a[22 * N + n] * b[27 * N + n];
    c[22 * N + n] = a[18 * N + n] * b[4 * N + n] + a[19 * N + n] * b[10 * N + n] + a[22 * N + n];
    c[23 * N + n] = a[20 * N + n] * b[17 * N + n] + a[23 * N + n];
    c[24 * N + n] = a[24 * N + n] * b[0 * N + n] + a[25 * N + n] * b[6 * N + n] + a[26 * N + n] * b[12 * N + n] +
                    a[28 * N + n] * b[24 * N + n];
    c[25 * N + n] = a[24 * N + n] * b[1 * N + n] + a[25 * N + n] * b[7 * N + n] + a[26 * N + n] * b[13 * N + n] +
                    a[28 * N + n] * b[25 * N + n];
    c[26 * N + n] = a[26 * N + n];
    c[27 * N + n] = a[24 * N + n] * b[3 * N + n] + a[25 * N + n] * b[9 * N + n] + a[26 * N + n] * b[15 * N + n] +
                    a[27 * N + n] + a[28 * N + n] * b[27 * N + n];
    c[28 * N + n] = a[24 * N + n] * b[4 * N + n] + a[25 * N + n] * b[10 * N + n] + a[28 * N + n];
    c[29 * N + n] = a[26 * N + n] * b[17 * N + n] + a[29 * N + n];
    c[30 * N + n] = a[30 * N + n] * b[0 * N + n] + a[31 * N + n] * b[6 * N + n] + a[32 * N + n] * b[12 * N + n] +
                    a[34 * N + n] * b[24 * N + n];
    c[31 * N + n] = a[30 * N + n] * b[1 * N + n] + a[31 * N + n] * b[7 * N + n] + a[32 * N + n] * b[13 * N + n] +
                    a[34 * N + n] * b[25 * N + n];
    c[32 * N + n] = a[32 * N + n];
    c[33 * N + n] = a[30 * N + n] * b[3 * N + n] + a[31 * N + n] * b[9 * N + n] + a[32 * N + n] * b[15 * N + n] +
                    a[33 * N + n] + a[34 * N + n] * b[27 * N + n];
    c[34 * N + n] = a[30 * N + n] * b[4 * N + n] + a[31 * N + n] * b[10 * N + n] + a[34 * N + n];
    c[35 * N + n] = a[32 * N + n] * b[17 * N + n] + a[35 * N + n];
  }

#ifdef UNUSED
  // this version does not assume to know which elements are 0 or 1, so it does the full multiplication
  void MultHelixPropFull(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += A.constAt(n, i, k) * B.constAt(n, k, j);
        }
      }
    }
  }

  // this version does not assume to know which elements are 0 or 1, so it does the full multiplication
  void MultHelixPropFull(const MPlexLL& A, const MPlexLL& B, MPlexLL& C) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += A.constAt(n, i, k) * B.constAt(n, k, j);
        }
      }
    }
  }

  // this version does not assume to know which elements are 0 or 1, so it does the full mupltiplication
  void MultHelixPropTranspFull(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += B.constAt(n, i, k) * A.constAt(n, j, k);
        }
      }
    }
  }

  // this version does not assume to know which elements are 0 or 1, so it does the full mupltiplication
  void MultHelixPropTranspFull(const MPlexLL& A, const MPlexLL& B, MPlexLL& C) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += B.constAt(n, i, k) * A.constAt(n, j, k);
        }
      }
    }
  }
#endif
}  // end unnamed namespace

//==============================================================================

namespace mkfit {

  void helixAtRFromIterativeCCSFullJac(const MPlexLV& inPar,
                                       const MPlexQI& inChg,
                                       const MPlexQF& msRad,
                                       MPlexLV& outPar,
                                       MPlexLL& errorProp,
                                       const int N_proc) {
    errorProp.setVal(0.f);
    MPlexLL errorPropTmp(0.f);   //initialize to zero
    MPlexLL errorPropSwap(0.f);  //initialize to zero

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      //initialize erroProp to identity matrix
      errorProp(n, 0, 0) = 1.f;
      errorProp(n, 1, 1) = 1.f;
      errorProp(n, 2, 2) = 1.f;
      errorProp(n, 3, 3) = 1.f;
      errorProp(n, 4, 4) = 1.f;
      errorProp(n, 5, 5) = 1.f;

      const float k = inChg.constAt(n, 0, 0) * 100.f / (-Const::sol * Config::Bfield);
      const float r = msRad.constAt(n, 0, 0);
      float r0 = hipo(inPar.constAt(n, 0, 0), inPar.constAt(n, 1, 0));

      if (std::abs(r - r0) < 0.0001f) {
        dprint_np(n, "distance less than 1mum, skip");
        continue;
      }

      const float ipt = inPar.constAt(n, 3, 0);
      const float phiin = inPar.constAt(n, 4, 0);
      const float theta = inPar.constAt(n, 5, 0);

      //set those that are 1. before iterations
      errorPropTmp(n, 2, 2) = 1.f;
      errorPropTmp(n, 3, 3) = 1.f;
      errorPropTmp(n, 4, 4) = 1.f;
      errorPropTmp(n, 5, 5) = 1.f;

      float cosah = 0., sinah = 0.;
      //no trig approx here, phi and theta can be large
      float cosP = std::cos(phiin), sinP = std::sin(phiin);
      const float cosT = std::cos(theta), sinT = std::sin(theta);
      float pxin = cosP / ipt;
      float pyin = sinP / ipt;

      CMS_UNROLL_LOOP_COUNT(Config::Niter)
      for (int i = 0; i < Config::Niter; ++i) {
        dprint_np(n,
                  std::endl
                      << "attempt propagation from r=" << r0 << " to r=" << r << std::endl
                      << "x=" << outPar.At(n, 0, 0) << " y=" << outPar.At(n, 1, 0) << " z=" << outPar.At(n, 2, 0)
                      << " px=" << std::cos(phiin) / ipt << " py=" << std::sin(phiin) / ipt
                      << " pz=" << 1.f / (ipt * tan(theta)) << " q=" << inChg.constAt(n, 0, 0) << std::endl);

        r0 = hipo(outPar.constAt(n, 0, 0), outPar.constAt(n, 1, 0));
        const float ialpha = (r - r0) * ipt / k;
        //alpha+=ialpha;

        if constexpr (Config::useTrigApprox) {
          sincos4(ialpha * 0.5f, sinah, cosah);
        } else {
          cosah = std::cos(ialpha * 0.5f);
          sinah = std::sin(ialpha * 0.5f);
        }
        const float cosa = 1.f - 2.f * sinah * sinah;
        const float sina = 2.f * sinah * cosah;

        //derivatives of alpha
        const float dadx = -outPar.At(n, 0, 0) * ipt / (k * r0);
        const float dady = -outPar.At(n, 1, 0) * ipt / (k * r0);
        const float dadipt = (r - r0) / k;

        outPar.At(n, 0, 0) = outPar.constAt(n, 0, 0) + 2.f * k * sinah * (pxin * cosah - pyin * sinah);
        outPar.At(n, 1, 0) = outPar.constAt(n, 1, 0) + 2.f * k * sinah * (pyin * cosah + pxin * sinah);
        const float pxinold = pxin;  //copy before overwriting
        pxin = pxin * cosa - pyin * sina;
        pyin = pyin * cosa + pxinold * sina;

        //need phi at origin, so this goes before redefining phi
        //no trig approx here, phi can be large
        cosP = std::cos(outPar.At(n, 4, 0));
        sinP = std::sin(outPar.At(n, 4, 0));

        outPar.At(n, 2, 0) = outPar.constAt(n, 2, 0) + k * ialpha * cosT / (ipt * sinT);
        outPar.At(n, 3, 0) = ipt;
        outPar.At(n, 4, 0) = outPar.constAt(n, 4, 0) + ialpha;
        outPar.At(n, 5, 0) = theta;

        errorPropTmp(n, 0, 0) = 1.f + k * (cosP * dadx * cosa - sinP * dadx * sina) / ipt;
        errorPropTmp(n, 0, 1) = k * (cosP * dady * cosa - sinP * dady * sina) / ipt;
        errorPropTmp(n, 0, 3) =
            k * (cosP * (ipt * dadipt * cosa - sina) + sinP * ((1.f - cosa) - ipt * dadipt * sina)) / (ipt * ipt);
        errorPropTmp(n, 0, 4) = -k * (sinP * sina + cosP * (1.f - cosa)) / ipt;

        errorPropTmp(n, 1, 0) = k * (sinP * dadx * cosa + cosP * dadx * sina) / ipt;
        errorPropTmp(n, 1, 1) = 1.f + k * (sinP * dady * cosa + cosP * dady * sina) / ipt;
        errorPropTmp(n, 1, 3) =
            k * (sinP * (ipt * dadipt * cosa - sina) + cosP * (ipt * dadipt * sina - (1.f - cosa))) / (ipt * ipt);
        errorPropTmp(n, 1, 4) = k * (cosP * sina - sinP * (1.f - cosa)) / ipt;

        errorPropTmp(n, 2, 0) = k * cosT * dadx / (ipt * sinT);
        errorPropTmp(n, 2, 1) = k * cosT * dady / (ipt * sinT);
        errorPropTmp(n, 2, 3) = k * cosT * (ipt * dadipt - ialpha) / (ipt * ipt * sinT);
        errorPropTmp(n, 2, 5) = -k * ialpha / (ipt * sinT * sinT);

        errorPropTmp(n, 4, 0) = dadx;
        errorPropTmp(n, 4, 1) = dady;
        errorPropTmp(n, 4, 3) = dadipt;

        MultHelixPropTemp(errorProp, errorPropTmp, errorPropSwap, n);
        errorProp = errorPropSwap;
      }

      dprint_np(
          n,
          "propagation end, dump parameters"
              << std::endl
              << "pos = " << outPar.At(n, 0, 0) << " " << outPar.At(n, 1, 0) << " " << outPar.At(n, 2, 0) << std::endl
              << "mom = " << std::cos(outPar.At(n, 4, 0)) / outPar.At(n, 3, 0) << " "
              << std::sin(outPar.At(n, 4, 0)) / outPar.At(n, 3, 0) << " "
              << 1. / (outPar.At(n, 3, 0) * tan(outPar.At(n, 5, 0)))
              << " r=" << std::sqrt(outPar.At(n, 0, 0) * outPar.At(n, 0, 0) + outPar.At(n, 1, 0) * outPar.At(n, 1, 0))
              << " pT=" << 1. / std::abs(outPar.At(n, 3, 0)) << std::endl);

#ifdef DEBUG
      if (debug && g_debug && n < N_proc) {
        dmutex_guard;
        std::cout << n << " jacobian" << std::endl;
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 0, 0),
               errorProp(n, 0, 1),
               errorProp(n, 0, 2),
               errorProp(n, 0, 3),
               errorProp(n, 0, 4),
               errorProp(n, 0, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 1, 0),
               errorProp(n, 1, 1),
               errorProp(n, 1, 2),
               errorProp(n, 1, 3),
               errorProp(n, 1, 4),
               errorProp(n, 1, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 2, 0),
               errorProp(n, 2, 1),
               errorProp(n, 2, 2),
               errorProp(n, 2, 3),
               errorProp(n, 2, 4),
               errorProp(n, 2, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 3, 0),
               errorProp(n, 3, 1),
               errorProp(n, 3, 2),
               errorProp(n, 3, 3),
               errorProp(n, 3, 4),
               errorProp(n, 3, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 4, 0),
               errorProp(n, 4, 1),
               errorProp(n, 4, 2),
               errorProp(n, 4, 3),
               errorProp(n, 4, 4),
               errorProp(n, 4, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 5, 0),
               errorProp(n, 5, 1),
               errorProp(n, 5, 2),
               errorProp(n, 5, 3),
               errorProp(n, 5, 4),
               errorProp(n, 5, 5));
      }
#endif
    }
  }

}  // end namespace mkfit

//#pragma omp declare simd simdlen(NN) notinbranch linear(n)
#include "PropagationMPlex.icc"

namespace mkfit {

  void helixAtRFromIterativeCCS(const MPlexLV& inPar,
                                const MPlexQI& inChg,
                                const MPlexQF& msRad,
                                MPlexLV& outPar,
                                MPlexLL& errorProp,
                                MPlexQI& outFailFlag,
                                const int N_proc,
                                const PropagationFlags& pflags) {
    errorProp.setVal(0.f);
    outFailFlag.setVal(0.f);

    helixAtRFromIterativeCCS_impl(inPar, inChg, msRad, outPar, errorProp, outFailFlag, 0, NN, N_proc, pflags);
  }

  void propagateHelixToRMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msRad,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags& pflags,
                              const MPlexQI* noMatEffPtr) {
    // bool debug = true;

    // This is used further down when calculating similarity with errorProp (and before in DEBUG).
    // MT: I don't think this really needed if we use inErr where required.
    outErr = inErr;
    // This requirement for helixAtRFromIterativeCCS_impl() and for helixAtRFromIterativeCCSFullJac().
    // MT: This should be properly handled in both functions (expecting input in output parameters sucks).
    outPar = inPar;

    MPlexLL errorProp;

    helixAtRFromIterativeCCS(inPar, inChg, msRad, outPar, errorProp, outFailFlag, N_proc, pflags);

#ifdef DEBUG
    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("outErr before prop %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", outErr.At(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");

        dprintf("errorProp %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", errorProp.At(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");
      }
    }
#endif

    if (pflags.apply_material) {
      MPlexQF hitsRl;
      MPlexQF hitsXi;
      MPlexQF propSign;

      const TrackerInfo& tinfo = *pflags.tracker_info;

#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        if (n >= N_proc || (outFailFlag(n, 0, 0) || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0)))) {
          hitsRl(n, 0, 0) = 0.f;
          hitsXi(n, 0, 0) = 0.f;
        } else {
          auto mat = tinfo.material_checked(std::abs(outPar(n, 2, 0)), msRad(n, 0, 0));
          hitsRl(n, 0, 0) = mat.radl;
          hitsXi(n, 0, 0) = mat.bbxi;
        }
        const float r0 = hipo(inPar(n, 0, 0), inPar(n, 1, 0));
        const float r = msRad(n, 0, 0);
        propSign(n, 0, 0) = (r > r0 ? 1. : -1.);
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, outErr, outPar, N_proc, true);
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);

    // MultHelixProp can be optimized for CCS coordinates, see GenMPlexOps.pl
    MPlexLL temp;
    MultHelixProp(errorProp, outErr, temp);
    MultHelixPropTransp(errorProp, temp, outErr);

    /*
     // To be used with: MPT_DIM = 1
     if (fabs(sqrt(outPar[0]*outPar[0]+outPar[1]*outPar[1]) - msRad[0]) > 0.0001)
     {
       std::cout << "DID NOT GET TO R, FailFlag=" << failFlag[0]
                 << " dR=" << msRad[0] - std::hypot(outPar[0],outPar[1])
                 << " r="  << msRad[0] << " rin=" << std::hypot(inPar[0],inPar[1]) << " rout=" << std::hypot(outPar[0],outPar[1])
                 << std::endl;
       // std::cout << "    pt=" << pt << " pz=" << inPar.At(n, 2) << std::endl;
     }
   */

    // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
    // state to input when propagation fails -- as was the default before.
    // if (pflags.copy_input_state_on_fail) {
    for (int i = 0; i < N_proc; ++i) {
      if (outFailFlag(i, 0, 0)) {
        outPar.copySlot(i, inPar);
        outErr.copySlot(i, inErr);
      }
    }
    // }
  }

  //==============================================================================

  void propagateHelixToZMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msZ,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags& pflags,
                              const MPlexQI* noMatEffPtr) {
    // debug = true;

    outErr = inErr;
    outPar = inPar;

    MPlexLL errorProp;

    helixAtZ(inPar, inChg, msZ, outPar, errorProp, outFailFlag, N_proc, pflags);

#ifdef DEBUG
    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("inErr %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", inErr.constAt(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");

        dprintf("errorProp %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", errorProp.At(kk, i, j));
          dprintf("\n");
        }
        dprintf("\n");
      }
    }
#endif

    if (pflags.apply_material) {
      MPlexQF hitsRl;
      MPlexQF hitsXi;
      MPlexQF propSign;

      const TrackerInfo& tinfo = *pflags.tracker_info;

#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        if (n >= N_proc || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
          hitsRl(n, 0, 0) = 0.f;
          hitsXi(n, 0, 0) = 0.f;
        } else {
          const float hypo = std::hypot(outPar(n, 0, 0), outPar(n, 1, 0));
          auto mat = tinfo.material_checked(std::abs(msZ(n, 0, 0)), hypo);
          hitsRl(n, 0, 0) = mat.radl;
          hitsXi(n, 0, 0) = mat.bbxi;
        }
        const float zout = msZ.constAt(n, 0, 0);
        const float zin = inPar.constAt(n, 2, 0);
        propSign(n, 0, 0) = (std::abs(zout) > std::abs(zin) ? 1. : -1.);
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, outErr, outPar, N_proc, false);
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);
    MPlexLL temp;
    MultHelixPropEndcap(errorProp, outErr, temp);
    MultHelixPropTranspEndcap(errorProp, temp, outErr);

    // PROP-FAIL-ENABLE To keep physics changes minimal, we always restore the
    // state to input when propagation fails -- as was the default before.
    // if (pflags.copy_input_state_on_fail) {
    for (int i = 0; i < N_proc; ++i) {
      if (outFailFlag(i, 0, 0)) {
        outPar.copySlot(i, inPar);
        outErr.copySlot(i, inErr);
      }
    }
    // }

    // This dump is now out of its place as similarity is done with matriplex ops.
    /*
#ifdef DEBUG
   {
     dmutex_guard;
     for (int kk = 0; kk < N_proc; ++kk)
     {
       dprintf("outErr %d\n", kk);
       for (int i = 0; i < 6; ++i) { for (int j = 0; j < 6; ++j)
           dprintf("%8f ", outErr.At(kk,i,j)); printf("\n");
       } dprintf("\n");

       dprintf("outPar %d\n", kk);
       for (int i = 0; i < 6; ++i) {
           dprintf("%8f ", outPar.At(kk,i,0)); printf("\n");
       } dprintf("\n");
       if (std::abs(outPar.At(kk,2,0) - msZ.constAt(kk, 0, 0)) > 0.0001) {
         float pt = 1.0f / inPar.constAt(kk,3,0);
	 dprint_np(kk, "DID NOT GET TO Z, dZ=" << std::abs(outPar.At(kk,2,0) - msZ.constAt(kk, 0, 0))
		   << " z=" << msZ.constAt(kk, 0, 0) << " zin=" << inPar.constAt(kk,2,0) << " zout=" << outPar.At(kk,2,0) << std::endl
		   << "pt=" << pt << " pz=" << pt/std::tan(inPar.constAt(kk,5,0)));
       }
     }
   }
#endif
   */
  }

  void helixAtZ(const MPlexLV& inPar,
                const MPlexQI& inChg,
                const MPlexQF& msZ,
                MPlexLV& outPar,
                MPlexLL& errorProp,
                MPlexQI& outFailFlag,
                const int N_proc,
                const PropagationFlags& pflags) {
    errorProp.setVal(0.f);

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      //initialize erroProp to identity matrix, except element 2,2 which is zero
      errorProp(n, 0, 0) = 1.f;
      errorProp(n, 1, 1) = 1.f;
      errorProp(n, 3, 3) = 1.f;
      errorProp(n, 4, 4) = 1.f;
      errorProp(n, 5, 5) = 1.f;
    }
    float zout[NN];
    float zin[NN];
    float ipt[NN];
    float phiin[NN];
    float theta[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      //initialize erroProp to identity matrix, except element 2,2 which is zero
      zout[n] = msZ.constAt(n, 0, 0);
      zin[n] = inPar.constAt(n, 2, 0);
      ipt[n] = inPar.constAt(n, 3, 0);
      phiin[n] = inPar.constAt(n, 4, 0);
      theta[n] = inPar.constAt(n, 5, 0);
    }

    float k[NN];
    if (pflags.use_param_b_field) {
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        k[n] = inChg.constAt(n, 0, 0) * 100.f /
               (-Const::sol * Config::bFieldFromZR(zin[n], hipo(inPar.constAt(n, 0, 0), inPar.constAt(n, 1, 0))));
      }
    } else {
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        k[n] = inChg.constAt(n, 0, 0) * 100.f / (-Const::sol * Config::Bfield);
      }
    }

    float kinv[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      kinv[n] = 1.f / k[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(n,
                std::endl
                    << "input parameters"
                    << " inPar.constAt(n, 0, 0)=" << std::setprecision(9) << inPar.constAt(n, 0, 0)
                    << " inPar.constAt(n, 1, 0)=" << std::setprecision(9) << inPar.constAt(n, 1, 0)
                    << " inPar.constAt(n, 2, 0)=" << std::setprecision(9) << inPar.constAt(n, 2, 0)
                    << " inPar.constAt(n, 3, 0)=" << std::setprecision(9) << inPar.constAt(n, 3, 0)
                    << " inPar.constAt(n, 4, 0)=" << std::setprecision(9) << inPar.constAt(n, 4, 0)
                    << " inPar.constAt(n, 5, 0)=" << std::setprecision(9) << inPar.constAt(n, 5, 0));
    }

    float pt[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pt[n] = 1.f / ipt[n];
    }

    //no trig approx here, phi can be large
    float cosP[NN];
    float sinP[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      cosP[n] = std::cos(phiin[n]);
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      sinP[n] = std::sin(phiin[n]);
    }

    float cosT[NN];
    float sinT[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      cosT[n] = std::cos(theta[n]);
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      sinT[n] = std::sin(theta[n]);
    }

    float tanT[NN];
    float icos2T[NN];
    float pxin[NN];
    float pyin[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      tanT[n] = sinT[n] / cosT[n];
      icos2T[n] = 1.f / (cosT[n] * cosT[n]);
      pxin[n] = cosP[n] * pt[n];
      pyin[n] = sinP[n] * pt[n];
    }
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      //fixme, make this printout useful for propagation to z
      dprint_np(n,
                std::endl
                    << "k=" << std::setprecision(9) << k[n] << " pxin=" << std::setprecision(9) << pxin[n]
                    << " pyin=" << std::setprecision(9) << pyin[n] << " cosP=" << std::setprecision(9) << cosP[n]
                    << " sinP=" << std::setprecision(9) << sinP[n] << " pt=" << std::setprecision(9) << pt[n]);
    }
    float deltaZ[NN];
    float alpha[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      deltaZ[n] = zout[n] - zin[n];
      alpha[n] = deltaZ[n] * tanT[n] * ipt[n] * kinv[n];
    }

    float cosahTmp[NN];
    float sinahTmp[NN];
    if constexpr (Config::useTrigApprox) {
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        sincos4(alpha[n] * 0.5f, sinahTmp[n], cosahTmp[n]);
      }
    } else {
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        cosahTmp[n] = std::cos(alpha[n] * 0.5f);
      }
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        sinahTmp[n] = std::sin(alpha[n] * 0.5f);
      }
    }

    float cosah[NN];
    float sinah[NN];
    float cosa[NN];
    float sina[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      cosah[n] = cosahTmp[n];
      sinah[n] = sinahTmp[n];
      cosa[n] = 1.f - 2.f * sinah[n] * sinah[n];
      sina[n] = 2.f * sinah[n] * cosah[n];
    }
//update parameters
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      outPar.At(n, 0, 0) = outPar.At(n, 0, 0) + 2.f * k[n] * sinah[n] * (pxin[n] * cosah[n] - pyin[n] * sinah[n]);
      outPar.At(n, 1, 0) = outPar.At(n, 1, 0) + 2.f * k[n] * sinah[n] * (pyin[n] * cosah[n] + pxin[n] * sinah[n]);
      outPar.At(n, 2, 0) = zout[n];
      outPar.At(n, 4, 0) = phiin[n] + alpha[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(n,
                std::endl
                    << "outPar.At(n, 0, 0)=" << outPar.At(n, 0, 0) << " outPar.At(n, 1, 0)=" << outPar.At(n, 1, 0)
                    << " pxin=" << pxin[n] << " pyin=" << pyin[n]);
    }

    float pxcaMpysa[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pxcaMpysa[n] = pxin[n] * cosa[n] - pyin[n] * sina[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      errorProp(n, 0, 2) = -tanT[n] * ipt[n] * pxcaMpysa[n];
      errorProp(n, 0, 3) =
          k[n] * pt[n] * pt[n] *
          (cosP[n] * (alpha[n] * cosa[n] - sina[n]) + sinP[n] * 2.f * sinah[n] * (sinah[n] - alpha[n] * cosah[n]));
      errorProp(n, 0, 4) = -2.f * k[n] * pt[n] * sinah[n] * (sinP[n] * cosah[n] + cosP[n] * sinah[n]);
      errorProp(n, 0, 5) = deltaZ[n] * ipt[n] * pxcaMpysa[n] * icos2T[n];
    }

    float pycaPpxsa[NN];
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      pycaPpxsa[n] = pyin[n] * cosa[n] + pxin[n] * sina[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      errorProp(n, 1, 2) = -tanT[n] * ipt[n] * pycaPpxsa[n];
      errorProp(n, 1, 3) =
          k[n] * pt[n] * pt[n] *
          (sinP[n] * (alpha[n] * cosa[n] - sina[n]) - cosP[n] * 2.f * sinah[n] * (sinah[n] - alpha[n] * cosah[n]));
      errorProp(n, 1, 4) = 2.f * k[n] * pt[n] * sinah[n] * (cosP[n] * cosah[n] - sinP[n] * sinah[n]);
      errorProp(n, 1, 5) = deltaZ[n] * ipt[n] * pycaPpxsa[n] * icos2T[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      errorProp(n, 4, 2) = -ipt[n] * tanT[n] * kinv[n];
      errorProp(n, 4, 3) = tanT[n] * deltaZ[n] * kinv[n];
      errorProp(n, 4, 5) = ipt[n] * deltaZ[n] * kinv[n] * icos2T[n];
    }

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      dprint_np(
          n,
          "propagation end, dump parameters"
              << std::endl
              << "pos = " << outPar.At(n, 0, 0) << " " << outPar.At(n, 1, 0) << " " << outPar.At(n, 2, 0) << std::endl
              << "mom = " << std::cos(outPar.At(n, 4, 0)) / outPar.At(n, 3, 0) << " "
              << std::sin(outPar.At(n, 4, 0)) / outPar.At(n, 3, 0) << " "
              << 1. / (outPar.At(n, 3, 0) * tan(outPar.At(n, 5, 0)))
              << " r=" << std::sqrt(outPar.At(n, 0, 0) * outPar.At(n, 0, 0) + outPar.At(n, 1, 0) * outPar.At(n, 1, 0))
              << " pT=" << 1. / std::abs(outPar.At(n, 3, 0)) << std::endl);
    }

    // PROP-FAIL-ENABLE Disabled to keep physics changes minimal.
    // To be reviewed, enabled and processed accordingly elsewhere.
    /*
    // Check for errors, set fail-flag.
    for (int n = 0; n < NN; ++n) {
      // We propagate for alpha: mark fail when prop angle more than pi/2
      if (std::abs(alpha[n]) > 1.57) {
        dprintf("helixAtZ: more than quarter turn, alpha = %f\n", alpha[n]);
        outFailFlag[n] = 1;
      } else {
        // Have we reached desired z? We can't know, we copy desired z to actual z.
        // Are we close to apex? Same condition as in propToR, 12.5 deg, cos(78.5deg) = 0.2
        float dotp = (outPar.At(n, 0, 0) * std::cos(outPar.At(n, 4, 0)) +
                      outPar.At(n, 1, 0) * std::sin(outPar.At(n, 4, 0))) /
                     std::hypot(outPar.At(n, 0, 0), outPar.At(n, 1, 0));
        if (dotp < 0.2 || dotp < 0) {
          dprintf("helixAtZ: dot product bad, dotp = %f\n", dotp);
          outFailFlag[n] = 1;
        }
      }
    }
    */

#ifdef DEBUG
    if (debug && g_debug) {
      for (int n = 0; n < N_proc; ++n) {
        dmutex_guard;
        std::cout << n << ": jacobian" << std::endl;
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 0, 0),
               errorProp(n, 0, 1),
               errorProp(n, 0, 2),
               errorProp(n, 0, 3),
               errorProp(n, 0, 4),
               errorProp(n, 0, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 1, 0),
               errorProp(n, 1, 1),
               errorProp(n, 1, 2),
               errorProp(n, 1, 3),
               errorProp(n, 1, 4),
               errorProp(n, 1, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 2, 0),
               errorProp(n, 2, 1),
               errorProp(n, 2, 2),
               errorProp(n, 2, 3),
               errorProp(n, 2, 4),
               errorProp(n, 2, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 3, 0),
               errorProp(n, 3, 1),
               errorProp(n, 3, 2),
               errorProp(n, 3, 3),
               errorProp(n, 3, 4),
               errorProp(n, 3, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 4, 0),
               errorProp(n, 4, 1),
               errorProp(n, 4, 2),
               errorProp(n, 4, 3),
               errorProp(n, 4, 4),
               errorProp(n, 4, 5));
        printf("%5f %5f %5f %5f %5f %5f\n",
               errorProp(n, 5, 0),
               errorProp(n, 5, 1),
               errorProp(n, 5, 2),
               errorProp(n, 5, 3),
               errorProp(n, 5, 4),
               errorProp(n, 5, 5));
      }
    }
#endif
  }

  //==============================================================================

  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            const int N_proc,
                            const bool isBarrel) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      float radL = hitsRl.constAt(n, 0, 0);
      if (radL < 1e-13f)
        continue;  //ugly, please fixme
      const float theta = outPar.constAt(n, 5, 0);
      const float pt = 1.f / outPar.constAt(n, 3, 0);  //fixme, make sure it is positive?
      const float p = pt / std::sin(theta);
      const float p2 = p * p;
      constexpr float mpi = 0.140;       // m=140 MeV, pion
      constexpr float mpi2 = mpi * mpi;  // m=140 MeV, pion
      const float beta2 = p2 / (p2 + mpi2);
      const float beta = std::sqrt(beta2);
      //radiation lenght, corrected for the crossing angle (cos alpha from dot product of radius vector and momentum)
      const float invCos = (isBarrel ? p / pt : 1.f / std::abs(std::cos(theta)));
      radL = radL * invCos;  //fixme works only for barrel geom
      // multiple scattering
      //vary independently phi and theta by the rms of the planar multiple scattering angle
      // XXX-KMD radL < 0, see your fixme above! Repeating bailout
      if (radL < 1e-13f)
        continue;
      // const float thetaMSC = 0.0136f*std::sqrt(radL)*(1.f+0.038f*std::log(radL))/(beta*p);// eq 32.15
      // const float thetaMSC2 = thetaMSC*thetaMSC;
      const float thetaMSC = 0.0136f * (1.f + 0.038f * std::log(radL)) / (beta * p);  // eq 32.15
      const float thetaMSC2 = thetaMSC * thetaMSC * radL;
      outErr.At(n, 4, 4) += thetaMSC2;
      // outErr.At(n, 4, 5) += thetaMSC2;
      outErr.At(n, 5, 5) += thetaMSC2;
      //std::cout << "beta=" << beta << " p=" << p << std::endl;
      //std::cout << "multiple scattering thetaMSC=" << thetaMSC << " thetaMSC2=" << thetaMSC2 << " radL=" << radL << std::endl;
      // energy loss
      // XXX-KMD beta2 = 1 => 1 / sqrt(0)
      // const float gamma = 1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      // const float gamma2 = gamma*gamma;
      const float gamma2 = (p2 + mpi2) / mpi2;
      const float gamma = std::sqrt(gamma2);  //1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      constexpr float me = 0.0005;            // m=0.5 MeV, electron
      const float wmax = 2.f * me * beta2 * gamma2 / (1.f + 2.f * gamma * me / mpi + me * me / (mpi * mpi));
      constexpr float I = 16.0e-9 * 10.75;
      const float deltahalf = std::log(28.816e-9f * std::sqrt(2.33f * 0.498f) / I) + std::log(beta * gamma) - 0.5f;
      const float dEdx =
          beta < 1.f
              ? (2.f * (hitsXi.constAt(n, 0, 0) * invCos *
                        (0.5f * std::log(2.f * me * beta2 * gamma2 * wmax / (I * I)) - beta2 - deltahalf) / beta2))
              : 0.f;  //protect against infs and nans
      // dEdx = dEdx*2.;//xi in cmssw is defined with an extra factor 0.5 with respect to formula 27.1 in pdg
      //std::cout << "dEdx=" << dEdx << " delta=" << deltahalf << " wmax=" << wmax << " Xi=" << hitsXi.constAt(n,0,0) << std::endl;
      const float dP = propSign.constAt(n, 0, 0) * dEdx / beta;
      outPar.At(n, 3, 0) = p / (std::max(p + dP, 0.001f) * pt);  //stay above 1MeV
      //assume 100% uncertainty
      outErr.At(n, 3, 3) += dP * dP / (p2 * pt * pt);
    }
  }

}  // end namespace mkfit
