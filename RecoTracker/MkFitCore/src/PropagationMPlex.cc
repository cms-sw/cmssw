#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "PropagationMPlex.h"

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  //==============================================================================
  // propagateLineToRMPlex
  //==============================================================================

  void propagateLineToRMPlex(const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             const int N_proc) {
    // XXX Regenerate parts below with a script.

    const Matriplex::idx_t N = NN;

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
    const Matriplex::idx_t N = NN;

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
    const Matriplex::idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const T* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    T* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "MultHelixPropTransp.ah"
  }

  void MultHelixPropTemp(const MPlexLL& A, const MPlexLL& B, MPlexLL& C, int n) {
    // C = A * B

    typedef float T;
    const Matriplex::idx_t N = NN;

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

    // loop does not vectorize with llvm16, and it issues a warning
    // that apparently can't be suppressed with a pragma.  Needs to
    // be rechecked if future llvm versions improve vectorization.
#if !defined(__clang__)
#pragma omp simd
#endif
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

// ============================================================================
// BEGIN STUFF FROM PropagationMPlex.icc

namespace {

  //========================================================================================
  // helixAtR
  //========================================================================================

  void helixAtRFromIterativeCCS_impl(const MPlexLV& __restrict__ inPar,
                                     const MPlexQI& __restrict__ inChg,
                                     const MPlexQF& __restrict__ msRad,
                                     MPlexLV& __restrict__ outPar,
                                     MPlexLL& __restrict__ errorProp,
                                     MPlexQI& __restrict__ outFailFlag,  // expected to be initialized to 0
                                     const int nmin,
                                     const int nmax,
                                     const int N_proc,
                                     const PropagationFlags& pf) {
    // bool debug = true;

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      //initialize erroProp to identity matrix
      errorProp(n, 0, 0) = 1.f;
      errorProp(n, 1, 1) = 1.f;
      errorProp(n, 2, 2) = 1.f;
      errorProp(n, 3, 3) = 1.f;
      errorProp(n, 4, 4) = 1.f;
      errorProp(n, 5, 5) = 1.f;
    }
    float r0[nmax - nmin];
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      //initialize erroProp to identity matrix
      r0[n - nmin] = hipo(inPar(n, 0, 0), inPar(n, 1, 0));
    }
    float k[nmax - nmin];
    if (pf.use_param_b_field) {
#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        k[n - nmin] = inChg(n, 0, 0) * 100.f / (-Const::sol * Config::bFieldFromZR(inPar(n, 2, 0), r0[n - nmin]));
      }
    } else {
#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        k[n - nmin] = inChg(n, 0, 0) * 100.f / (-Const::sol * Config::Bfield);
      }
    }
    float r[nmax - nmin];
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      r[n - nmin] = msRad(n, 0, 0);
    }
    float xin[nmax - nmin];
    float yin[nmax - nmin];
    float ipt[nmax - nmin];
    float phiin[nmax - nmin];
    float theta[nmax - nmin];
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      // if (std::abs(r-r0)<0.0001f) {
      // 	dprint("distance less than 1mum, skip");
      // 	continue;
      // }

      xin[n - nmin] = inPar(n, 0, 0);
      yin[n - nmin] = inPar(n, 1, 0);
      ipt[n - nmin] = inPar(n, 3, 0);
      phiin[n - nmin] = inPar(n, 4, 0);
      theta[n - nmin] = inPar(n, 5, 0);

      //dprint(std::endl);
    }

    //debug = true;
    for (int n = nmin; n < nmax; ++n) {
      dprint_np(n,
                "input parameters"
                    << " inPar(n, 0, 0)=" << std::setprecision(9) << inPar(n, 0, 0) << " inPar(n, 1, 0)="
                    << std::setprecision(9) << inPar(n, 1, 0) << " inPar(n, 2, 0)=" << std::setprecision(9)
                    << inPar(n, 2, 0) << " inPar(n, 3, 0)=" << std::setprecision(9) << inPar(n, 3, 0)
                    << " inPar(n, 4, 0)=" << std::setprecision(9) << inPar(n, 4, 0)
                    << " inPar(n, 5, 0)=" << std::setprecision(9) << inPar(n, 5, 0));
    }

    float kinv[nmax - nmin];
    float pt[nmax - nmin];
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      kinv[n - nmin] = 1.f / k[n - nmin];
      pt[n - nmin] = 1.f / ipt[n - nmin];
    }
    float D[nmax - nmin];
    float cosa[nmax - nmin];
    float sina[nmax - nmin];
    float cosah[nmax - nmin];
    float sinah[nmax - nmin];
    float id[nmax - nmin];

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      D[n - nmin] = 0.;
    }

    //no trig approx here, phi can be large
    float cosPorT[nmax - nmin];
    float sinPorT[nmax - nmin];
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      cosPorT[n - nmin] = std::cos(phiin[n - nmin]);
    }
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      sinPorT[n - nmin] = std::sin(phiin[n - nmin]);
    }

    float pxin[nmax - nmin];
    float pyin[nmax - nmin];
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      pxin[n - nmin] = cosPorT[n - nmin] * pt[n - nmin];
      pyin[n - nmin] = sinPorT[n - nmin] * pt[n - nmin];
    }

    for (int n = nmin; n < nmax; ++n) {
      dprint_np(n,
                "k=" << std::setprecision(9) << k[n - nmin] << " pxin=" << std::setprecision(9) << pxin[n - nmin]
                     << " pyin=" << std::setprecision(9) << pyin[n - nmin] << " cosPorT=" << std::setprecision(9)
                     << cosPorT[n - nmin] << " sinPorT=" << std::setprecision(9) << sinPorT[n - nmin]
                     << " pt=" << std::setprecision(9) << pt[n - nmin]);
    }

    float dDdx[nmax - nmin];
    float dDdy[nmax - nmin];
    float dDdipt[nmax - nmin];
    float dDdphi[nmax - nmin];

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      dDdipt[n - nmin] = 0.;
      dDdphi[n - nmin] = 0.;
    }
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      //derivatives initialized to value for first iteration, i.e. distance = r-r0in
      dDdx[n - nmin] = r0[n - nmin] > 0.f ? -xin[n - nmin] / r0[n - nmin] : 0.f;
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      dDdy[n - nmin] = r0[n - nmin] > 0.f ? -yin[n - nmin] / r0[n - nmin] : 0.f;
    }

    float oodotp[nmax - nmin];
    float x[nmax - nmin];
    float y[nmax - nmin];
    float oor0[nmax - nmin];
    float dadipt[nmax - nmin];
    float dadx[nmax - nmin];
    float dady[nmax - nmin];
    float pxca[nmax - nmin];
    float pxsa[nmax - nmin];
    float pyca[nmax - nmin];
    float pysa[nmax - nmin];
    float tmp[nmax - nmin];
    float tmpx[nmax - nmin];
    float tmpy[nmax - nmin];
    float pxinold[nmax - nmin];

    CMS_UNROLL_LOOP_COUNT(Config::Niter)
    for (int i = 0; i < Config::Niter; ++i) {
#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        //compute distance and path for the current iteration
        r0[n - nmin] = hipo(outPar(n, 0, 0), outPar(n, 1, 0));
      }

      // Use one over dot product of transverse momentum and radial
      // direction to scale the step. Propagation is prevented from reaching
      // too close to the apex (dotp > 0.2).
      // - Can / should we come up with a better approximation?
      // - Can / should take +/- curvature into account?

#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        oodotp[n - nmin] =
            r0[n - nmin] * pt[n - nmin] / (pxin[n - nmin] * outPar(n, 0, 0) + pyin[n - nmin] * outPar(n, 1, 0));
      }

#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        if (oodotp[n - nmin] > 5.0f || oodotp[n - nmin] < 0)  // 0.2 is 78.5 deg
        {
          outFailFlag(n, 0, 0) = 1;
          oodotp[n - nmin] = 0.0f;
        } else if (r[n - nmin] - r0[n - nmin] < 0.0f && pt[n - nmin] < 1.0f) {
          // Scale down the correction for low-pT ingoing tracks.
          oodotp[n - nmin] = 1.0f + (oodotp[n - nmin] - 1.0f) * pt[n - nmin];
        }
      }

#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        // Can we come up with a better approximation?
        // Should take +/- curvature into account.
        id[n - nmin] = (r[n - nmin] - r0[n - nmin]) * oodotp[n - nmin];
      }

#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        D[n - nmin] += id[n - nmin];
      }

      if constexpr (Config::useTrigApprox) {
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
        for (int n = nmin; n < nmax; ++n) {
          sincos4(id[n - nmin] * ipt[n - nmin] * kinv[n - nmin] * 0.5f, sinah[n - nmin], cosah[n - nmin]);
        }
      } else {
#if !defined(__INTEL_COMPILER)
#pragma omp simd
#endif
        for (int n = nmin; n < nmax; ++n) {
          cosah[n - nmin] = std::cos(id[n - nmin] * ipt[n - nmin] * kinv[n - nmin] * 0.5f);
          sinah[n - nmin] = std::sin(id[n - nmin] * ipt[n - nmin] * kinv[n - nmin] * 0.5f);
        }
      }

#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        cosa[n - nmin] = 1.f - 2.f * sinah[n - nmin] * sinah[n - nmin];
        sina[n - nmin] = 2.f * sinah[n - nmin] * cosah[n - nmin];
      }

      for (int n = nmin; n < nmax; ++n) {
        dprint_np(n,
                  "Attempt propagation from r="
                      << r0[n - nmin] << " to r=" << r[n - nmin] << std::endl
                      << "   x=" << xin[n - nmin] << " y=" << yin[n - nmin] << " z=" << inPar(n, 2, 0)
                      << " px=" << pxin[n - nmin] << " py=" << pyin[n - nmin]
                      << " pz=" << pt[n - nmin] * std::tan(theta[n - nmin]) << " q=" << inChg(n, 0, 0) << std::endl
                      << "   r=" << std::setprecision(9) << r[n - nmin] << " r0=" << std::setprecision(9)
                      << r0[n - nmin] << " id=" << std::setprecision(9) << id[n - nmin]
                      << " dr=" << std::setprecision(9) << r[n - nmin] - r0[n - nmin] << " cosa=" << cosa[n - nmin]
                      << " sina=" << sina[n - nmin] << " dir_cos(rad,pT)=" << 1.0f / oodotp[n - nmin]);
      }

      //update derivatives on total distance
      if (i + 1 != Config::Niter) {
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          x[n - nmin] = outPar(n, 0, 0);
          y[n - nmin] = outPar(n, 1, 0);
        }
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          oor0[n - nmin] =
              (r0[n - nmin] > 0.f && std::abs(r[n - nmin] - r0[n - nmin]) > 0.0001f) ? 1.f / r0[n - nmin] : 0.f;
        }
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          dadipt[n - nmin] = id[n - nmin] * kinv[n - nmin];
          dadx[n - nmin] = -x[n - nmin] * ipt[n - nmin] * kinv[n - nmin] * oor0[n - nmin];
          dady[n - nmin] = -y[n - nmin] * ipt[n - nmin] * kinv[n - nmin] * oor0[n - nmin];
          pxca[n - nmin] = pxin[n - nmin] * cosa[n - nmin];
          pxsa[n - nmin] = pxin[n - nmin] * sina[n - nmin];
          pyca[n - nmin] = pyin[n - nmin] * cosa[n - nmin];
          pysa[n - nmin] = pyin[n - nmin] * sina[n - nmin];
          tmpx[n - nmin] = k[n - nmin] * dadx[n - nmin];
        }

#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          dDdx[n - nmin] -= (x[n - nmin] * (1.f + tmpx[n - nmin] * (pxca[n - nmin] - pysa[n - nmin])) +
                             y[n - nmin] * tmpx[n - nmin] * (pyca[n - nmin] + pxsa[n - nmin])) *
                            oor0[n - nmin];
        }

#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          tmpy[n - nmin] = k[n - nmin] * dady[n - nmin];
        }
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          dDdy[n - nmin] -= (x[n - nmin] * tmpy[n - nmin] * (pxca[n - nmin] - pysa[n - nmin]) +
                             y[n - nmin] * (1.f + tmpy[n - nmin] * (pyca[n - nmin] + pxsa[n - nmin]))) *
                            oor0[n - nmin];
        }
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          //now r0 depends on ipt and phi as well
          tmp[n - nmin] = dadipt[n - nmin] * ipt[n - nmin];
        }
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          dDdipt[n - nmin] -= k[n - nmin] *
                              (x[n - nmin] * (pxca[n - nmin] * tmp[n - nmin] - pysa[n - nmin] * tmp[n - nmin] -
                                              pyca[n - nmin] - pxsa[n - nmin] + pyin[n - nmin]) +
                               y[n - nmin] * (pyca[n - nmin] * tmp[n - nmin] + pxsa[n - nmin] * tmp[n - nmin] -
                                              pysa[n - nmin] + pxca[n - nmin] - pxin[n - nmin])) *
                              pt[n - nmin] * oor0[n - nmin];
        }
#pragma omp simd
        for (int n = nmin; n < nmax; ++n) {
          dDdphi[n - nmin] += k[n - nmin] *
                              (x[n - nmin] * (pysa[n - nmin] - pxin[n - nmin] + pxca[n - nmin]) -
                               y[n - nmin] * (pxsa[n - nmin] - pyin[n - nmin] + pyca[n - nmin])) *
                              oor0[n - nmin];
        }
      }

#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        //update parameters
        outPar(n, 0, 0) = outPar(n, 0, 0) + 2.f * k[n - nmin] * sinah[n - nmin] *
                                                (pxin[n - nmin] * cosah[n - nmin] - pyin[n - nmin] * sinah[n - nmin]);
        outPar(n, 1, 0) = outPar(n, 1, 0) + 2.f * k[n - nmin] * sinah[n - nmin] *
                                                (pyin[n - nmin] * cosah[n - nmin] + pxin[n - nmin] * sinah[n - nmin]);
        pxinold[n - nmin] = pxin[n - nmin];  //copy before overwriting
        pxin[n - nmin] = pxin[n - nmin] * cosa[n - nmin] - pyin[n - nmin] * sina[n - nmin];
        pyin[n - nmin] = pyin[n - nmin] * cosa[n - nmin] + pxinold[n - nmin] * sina[n - nmin];
      }
      for (int n = nmin; n < nmax; ++n) {
        dprint_np(n,
                  "outPar(n, 0, 0)=" << outPar(n, 0, 0) << " outPar(n, 1, 0)=" << outPar(n, 1, 0)
                                     << " pxin=" << pxin[n - nmin] << " pyin=" << pyin[n - nmin]);
      }
    }  // iteration loop

    float alpha[nmax - nmin];
    float dadphi[nmax - nmin];

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      alpha[n - nmin] = D[n - nmin] * ipt[n - nmin] * kinv[n - nmin];
      dadx[n - nmin] = dDdx[n - nmin] * ipt[n - nmin] * kinv[n - nmin];
      dady[n - nmin] = dDdy[n - nmin] * ipt[n - nmin] * kinv[n - nmin];
      dadipt[n - nmin] = (ipt[n - nmin] * dDdipt[n - nmin] + D[n - nmin]) * kinv[n - nmin];
      dadphi[n - nmin] = dDdphi[n - nmin] * ipt[n - nmin] * kinv[n - nmin];
    }

    if constexpr (Config::useTrigApprox) {
#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        sincos4(alpha[n - nmin], sina[n - nmin], cosa[n - nmin]);
      }
    } else {
#pragma omp simd
      for (int n = nmin; n < nmax; ++n) {
        cosa[n - nmin] = std::cos(alpha[n - nmin]);
        sina[n - nmin] = std::sin(alpha[n - nmin]);
      }
    }
#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      errorProp(n, 0, 0) = 1.f + k[n - nmin] * dadx[n - nmin] *
                                     (cosPorT[n - nmin] * cosa[n - nmin] - sinPorT[n - nmin] * sina[n - nmin]) *
                                     pt[n - nmin];
      errorProp(n, 0, 1) = k[n - nmin] * dady[n - nmin] *
                           (cosPorT[n - nmin] * cosa[n - nmin] - sinPorT[n - nmin] * sina[n - nmin]) * pt[n - nmin];
      errorProp(n, 0, 2) = 0.f;
      errorProp(n, 0, 3) =
          k[n - nmin] *
          (cosPorT[n - nmin] * (ipt[n - nmin] * dadipt[n - nmin] * cosa[n - nmin] - sina[n - nmin]) +
           sinPorT[n - nmin] * ((1.f - cosa[n - nmin]) - ipt[n - nmin] * dadipt[n - nmin] * sina[n - nmin])) *
          pt[n - nmin] * pt[n - nmin];
      errorProp(n, 0, 4) = k[n - nmin] *
                           (cosPorT[n - nmin] * dadphi[n - nmin] * cosa[n - nmin] -
                            sinPorT[n - nmin] * dadphi[n - nmin] * sina[n - nmin] - sinPorT[n - nmin] * sina[n - nmin] +
                            cosPorT[n - nmin] * cosa[n - nmin] - cosPorT[n - nmin]) *
                           pt[n - nmin];
      errorProp(n, 0, 5) = 0.f;
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      errorProp(n, 1, 0) = k[n - nmin] * dadx[n - nmin] *
                           (sinPorT[n - nmin] * cosa[n - nmin] + cosPorT[n - nmin] * sina[n - nmin]) * pt[n - nmin];
      errorProp(n, 1, 1) = 1.f + k[n - nmin] * dady[n - nmin] *
                                     (sinPorT[n - nmin] * cosa[n - nmin] + cosPorT[n - nmin] * sina[n - nmin]) *
                                     pt[n - nmin];
      errorProp(n, 1, 2) = 0.f;
      errorProp(n, 1, 3) =
          k[n - nmin] *
          (sinPorT[n - nmin] * (ipt[n - nmin] * dadipt[n - nmin] * cosa[n - nmin] - sina[n - nmin]) +
           cosPorT[n - nmin] * (ipt[n - nmin] * dadipt[n - nmin] * sina[n - nmin] - (1.f - cosa[n - nmin]))) *
          pt[n - nmin] * pt[n - nmin];
      errorProp(n, 1, 4) = k[n - nmin] *
                           (sinPorT[n - nmin] * dadphi[n - nmin] * cosa[n - nmin] +
                            cosPorT[n - nmin] * dadphi[n - nmin] * sina[n - nmin] + sinPorT[n - nmin] * cosa[n - nmin] +
                            cosPorT[n - nmin] * sina[n - nmin] - sinPorT[n - nmin]) *
                           pt[n - nmin];
      errorProp(n, 1, 5) = 0.f;
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      //no trig approx here, theta can be large
      cosPorT[n - nmin] = std::cos(theta[n - nmin]);
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      sinPorT[n - nmin] = std::sin(theta[n - nmin]);
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      //redefine sinPorT as 1./sinPorT to reduce the number of temporaries
      sinPorT[n - nmin] = 1.f / sinPorT[n - nmin];
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      outPar(n, 2, 0) =
          inPar(n, 2, 0) + k[n - nmin] * alpha[n - nmin] * cosPorT[n - nmin] * pt[n - nmin] * sinPorT[n - nmin];
      errorProp(n, 2, 0) = k[n - nmin] * cosPorT[n - nmin] * dadx[n - nmin] * pt[n - nmin] * sinPorT[n - nmin];
      errorProp(n, 2, 1) = k[n - nmin] * cosPorT[n - nmin] * dady[n - nmin] * pt[n - nmin] * sinPorT[n - nmin];
      errorProp(n, 2, 2) = 1.f;
      errorProp(n, 2, 3) = k[n - nmin] * cosPorT[n - nmin] * (ipt[n - nmin] * dadipt[n - nmin] - alpha[n - nmin]) *
                           pt[n - nmin] * pt[n - nmin] * sinPorT[n - nmin];
      errorProp(n, 2, 4) = k[n - nmin] * dadphi[n - nmin] * cosPorT[n - nmin] * pt[n - nmin] * sinPorT[n - nmin];
      errorProp(n, 2, 5) = -k[n - nmin] * alpha[n - nmin] * pt[n - nmin] * sinPorT[n - nmin] * sinPorT[n - nmin];
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      outPar(n, 3, 0) = ipt[n - nmin];
      errorProp(n, 3, 0) = 0.f;
      errorProp(n, 3, 1) = 0.f;
      errorProp(n, 3, 2) = 0.f;
      errorProp(n, 3, 3) = 1.f;
      errorProp(n, 3, 4) = 0.f;
      errorProp(n, 3, 5) = 0.f;
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      outPar(n, 4, 0) = inPar(n, 4, 0) + alpha[n - nmin];
      errorProp(n, 4, 0) = dadx[n - nmin];
      errorProp(n, 4, 1) = dady[n - nmin];
      errorProp(n, 4, 2) = 0.f;
      errorProp(n, 4, 3) = dadipt[n - nmin];
      errorProp(n, 4, 4) = 1.f + dadphi[n - nmin];
      errorProp(n, 4, 5) = 0.f;
    }

#pragma omp simd
    for (int n = nmin; n < nmax; ++n) {
      outPar(n, 5, 0) = theta[n - nmin];
      errorProp(n, 5, 0) = 0.f;
      errorProp(n, 5, 1) = 0.f;
      errorProp(n, 5, 2) = 0.f;
      errorProp(n, 5, 3) = 0.f;
      errorProp(n, 5, 4) = 0.f;
      errorProp(n, 5, 5) = 1.f;
    }

    for (int n = nmin; n < nmax; ++n) {
      dprint_np(
          n,
          "propagation end, dump parameters\n"
              << "   D = " << D[n - nmin] << " alpha = " << alpha[n - nmin] << " kinv = " << kinv[n - nmin] << std::endl
              << "   pos = " << outPar(n, 0, 0) << " " << outPar(n, 1, 0) << " " << outPar(n, 2, 0) << "\t\t r="
              << std::sqrt(outPar(n, 0, 0) * outPar(n, 0, 0) + outPar(n, 1, 0) * outPar(n, 1, 0)) << std::endl
              << "   mom = " << outPar(n, 3, 0) << " " << outPar(n, 4, 0) << " " << outPar(n, 5, 0) << std::endl
              << "   cart= " << std::cos(outPar(n, 4, 0)) / outPar(n, 3, 0) << " "
              << std::sin(outPar(n, 4, 0)) / outPar(n, 3, 0) << " " << 1. / (outPar(n, 3, 0) * tan(outPar(n, 5, 0)))
              << "\t\tpT=" << 1. / std::abs(outPar(n, 3, 0)) << std::endl);
    }

#ifdef DEBUG
    for (int n = nmin; n < nmax; ++n) {
      if (debug && g_debug && n < N_proc) {
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
        printf("\n");
      }
    }
#endif
  }

}  // namespace

// END STUFF FROM PropagationMPlex.icc
// ============================================================================

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

    // MultHelixProp can be optimized for CCS coordinates, see GenMPlexOps.pl
    MPlexLL temp;
    MultHelixProp(errorProp, outErr, temp);
    MultHelixPropTransp(errorProp, temp, outErr);
    // can replace with: MultHelixPropFull(errorProp, outErr, temp); MultHelixPropTranspFull(errorProp, temp, outErr);

#ifdef DEBUG
    if (debug && g_debug) {
      for (int kk = 0; kk < N_proc; ++kk) {
        dprintf("outErr %d\n", kk);
        for (int i = 0; i < 6; ++i) {
          for (int j = 0; j < 6; ++j)
            dprintf("%8f ", outErr.constAt(kk, i, j));
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

#if !defined(__clang__)
#pragma omp simd
#endif
      for (int n = 0; n < NN; ++n) {
        if (n < N_proc) {
          if (outFailFlag(n, 0, 0) || (noMatEffPtr && noMatEffPtr->constAt(n, 0, 0))) {
            hitsRl(n, 0, 0) = 0.f;
            hitsXi(n, 0, 0) = 0.f;
          } else {
            const auto mat = tinfo.material_checked(std::abs(outPar(n, 2, 0)), msRad(n, 0, 0));
            hitsRl(n, 0, 0) = mat.radl;
            hitsXi(n, 0, 0) = mat.bbxi;
          }
          const float r0 = hipo(inPar(n, 0, 0), inPar(n, 1, 0));
          const float r = msRad(n, 0, 0);
          propSign(n, 0, 0) = (r > r0 ? 1.f : -1.f);
        }
      }
      MPlexHV plNrm;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        plNrm(n, 0, 0) = std::cos(outPar.constAt(n, 4, 0));
        plNrm(n, 1, 0) = std::sin(outPar.constAt(n, 4, 0));
        plNrm(n, 2, 0) = 0.f;
      }
      applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, outErr, outPar, N_proc);
#ifdef DEBUG
      if (debug && g_debug) {
        for (int kk = 0; kk < N_proc; ++kk) {
          dprintf("propSign %d\n", kk);
          for (int i = 0; i < 1; ++i) {
            dprintf("%8f ", propSign.constAt(kk, i, 0));
          }
          dprintf("\n");
          dprintf("plNrm %d\n", kk);
          for (int i = 0; i < 3; ++i) {
            dprintf("%8f ", plNrm.constAt(kk, i, 0));
          }
          dprintf("\n");
          dprintf("outErr(after material) %d\n", kk);
          for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j)
              dprintf("%8f ", outErr.constAt(kk, i, j));
            dprintf("\n");
          }
          dprintf("\n");
        }
      }
#endif
    }

    squashPhiMPlex(outPar, N_proc);  // ensure phi is between |pi|

    // Matriplex version of:
    // result.errors = ROOT::Math::Similarity(errorProp, outErr);

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

}  // end namespace mkfit
