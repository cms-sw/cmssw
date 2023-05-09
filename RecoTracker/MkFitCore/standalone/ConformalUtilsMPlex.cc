#include "ConformalUtilsMPlex.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"

//#define DEBUG
#include "RecoTracker/MkFitCore/src/Debug.h"

/* From MkFitter.h/.cc
// ----------------
  void ConformalFitTracks(bool fitting, int beg, int end);
// ----------------
  void MkFitter::ConformalFitTracks(bool fitting, int beg, int end) {
    // bool fitting to determine to use fitting CF error widths
    // in reality, this is depedent on hits used to make pulls
    // could consider writing an array for widths for a given hit combo
    // to give precise widths --> then would drop boolean
    // also used to determine which hits to use

    int front, middle, back;

    // FIXME FITTING HITS --> assume one hit per layer and all layers found! BAD! Need vector of indices to do this right instead...
    // can always assume 0,1,2 for seeding --> triplets in forward direction
#ifdef INWARDFIT
    front = (fitting ? Config::nLayers - 1
                     : 0);  // i.e. would rather have true option not hardcoded... but set by ACTUAL last hit found
    middle =
        (fitting ? (Config::nLayers - 1) / 2 : 1);  // same with this one... would rather middle hit be in the middle!
    back = (fitting ? 0 : 2);
#else
    front = (fitting ? 0 : 0);
    middle = (fitting ? (Config::nLayers - 1) / 2 : 1);  // ditto above
    back = (fitting ? Config::nLayers - 1 : 2);          // yup...
#endif

    // write to iC --> next step will be a propagation no matter what
    conformalFitMPlex(fitting, Label, Err[iC], Par[iC], msPar[front], msPar[middle], msPar[back]);

    // need to set most off-diagonal elements in unc. to zero, inflate all other elements;
    if (fitting) {
      using idx_t = Matriplex::idx_t;
      const idx_t N = NN;
#pragma omp simd
      for (int n = 0; n < N; ++n) {
        Err[iC].At(n, 0, 0) = Err[iC].constAt(n, 0, 0) * Config::blowupfit;
        Err[iC].At(n, 0, 1) = Err[iC].constAt(n, 0, 1) * Config::blowupfit;
        Err[iC].At(n, 1, 0) = Err[iC].constAt(n, 1, 0) * Config::blowupfit;
        Err[iC].At(n, 1, 1) = Err[iC].constAt(n, 1, 1) * Config::blowupfit;
        Err[iC].At(n, 2, 2) = Err[iC].constAt(n, 2, 2) * Config::blowupfit;
        Err[iC].At(n, 3, 3) = Err[iC].constAt(n, 3, 3) * Config::blowupfit;
        Err[iC].At(n, 4, 4) = Err[iC].constAt(n, 4, 4) * Config::blowupfit;
        Err[iC].At(n, 5, 5) = Err[iC].constAt(n, 5, 5) * Config::blowupfit;

        Err[iC].At(n, 0, 2) = 0.0f;
        Err[iC].At(n, 0, 3) = 0.0f;
        Err[iC].At(n, 0, 4) = 0.0f;
        Err[iC].At(n, 0, 5) = 0.0f;
        Err[iC].At(n, 1, 2) = 0.0f;
        Err[iC].At(n, 1, 3) = 0.0f;
        Err[iC].At(n, 1, 4) = 0.0f;
        Err[iC].At(n, 1, 5) = 0.0f;
        Err[iC].At(n, 2, 0) = 0.0f;
        Err[iC].At(n, 2, 1) = 0.0f;
        Err[iC].At(n, 2, 3) = 0.0f;
        Err[iC].At(n, 2, 4) = 0.0f;
        Err[iC].At(n, 2, 5) = 0.0f;
        Err[iC].At(n, 3, 0) = 0.0f;
        Err[iC].At(n, 3, 1) = 0.0f;
        Err[iC].At(n, 3, 2) = 0.0f;
        Err[iC].At(n, 3, 4) = 0.0f;
        Err[iC].At(n, 3, 5) = 0.0f;
        Err[iC].At(n, 4, 0) = 0.0f;
        Err[iC].At(n, 4, 1) = 0.0f;
        Err[iC].At(n, 4, 2) = 0.0f;
        Err[iC].At(n, 4, 3) = 0.0f;
        Err[iC].At(n, 4, 5) = 0.0f;
        Err[iC].At(n, 5, 0) = 0.0f;
        Err[iC].At(n, 5, 1) = 0.0f;
        Err[iC].At(n, 5, 2) = 0.0f;
        Err[iC].At(n, 5, 3) = 0.0f;
        Err[iC].At(n, 5, 4) = 0.0f;
      }
    }
  }
*/

namespace mkfit {

  inline void CFMap(const MPlexHH& A, const MPlexHV& B, MPlexHV& C) {
    using idx_t = Matriplex::idx_t;

    // C = A * B, C is 3x1, A is 3x3 , B is 3x1

    typedef float T;
    typedef float Tv;
    const idx_t N = NN;

    const T* a = A.fArray;
    ASSUME_ALIGNED(a, 64);
    const Tv* b = B.fArray;
    ASSUME_ALIGNED(b, 64);
    Tv* c = C.fArray;
    ASSUME_ALIGNED(c, 64);

#include "RecoTracker/MkFitCore/standalone/CFMatrix33Vector3.ah"
  }

  //M. Hansroul, H. Jeremie and D. Savard, NIM A 270 (1988) 498
  //http://www.sciencedirect.com/science/article/pii/016890028890722X

  void conformalFitMPlex(bool fitting,
                         MPlexQI seedID,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         const MPlexHV& msPar0,
                         const MPlexHV& msPar1,
                         const MPlexHV& msPar2) {
    bool debug(false);

    using idx_t = Matriplex::idx_t;
    const idx_t N = NN;

    // Store positions in mplex vectors... could consider storing in a 3x3 matrix, too
    MPlexHV x, y, z, r2;
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      x.At(n, 0, 0) = msPar0.constAt(n, 0, 0);
      x.At(n, 1, 0) = msPar1.constAt(n, 0, 0);
      x.At(n, 2, 0) = msPar2.constAt(n, 0, 0);

      y.At(n, 0, 0) = msPar0.constAt(n, 1, 0);
      y.At(n, 1, 0) = msPar1.constAt(n, 1, 0);
      y.At(n, 2, 0) = msPar2.constAt(n, 1, 0);

      z.At(n, 0, 0) = msPar0.constAt(n, 2, 0);
      z.At(n, 1, 0) = msPar1.constAt(n, 2, 0);
      z.At(n, 2, 0) = msPar2.constAt(n, 2, 0);

      for (int i = 0; i < 3; ++i) {
        r2.At(n, i, 0) = getRad2(x.constAt(n, i, 0), y.constAt(n, i, 0));
      }
    }

    // Start setting the output parameters
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      outPar.At(n, 0, 0) = x.constAt(n, 0, 0);
      outPar.At(n, 1, 0) = y.constAt(n, 0, 0);
      outPar.At(n, 2, 0) = z.constAt(n, 0, 0);
    }

    // Use r-phi smearing to set initial error estimation for positions
    // trackStates already initialized to identity for seeding ... don't store off-diag 0's, zero's for fitting set outside CF
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      const float varPhi = Config::varXY / r2.constAt(n, 0, 0);
      const float invvarR2 = Config::varR / r2.constAt(n, 0, 0);

      outErr.At(n, 0, 0) =
          x.constAt(n, 0, 0) * x.constAt(n, 0, 0) * invvarR2 + y.constAt(n, 0, 0) * y.constAt(n, 0, 0) * varPhi;
      outErr.At(n, 0, 1) = x.constAt(n, 0, 0) * y.constAt(n, 0, 0) * (invvarR2 - varPhi);

      outErr.At(n, 1, 0) = outErr.constAt(n, 0, 1);
      outErr.At(n, 1, 1) =
          y.constAt(n, 0, 0) * y.constAt(n, 0, 0) * invvarR2 + x.constAt(n, 0, 0) * x.constAt(n, 0, 0) * varPhi;

      outErr.At(n, 2, 2) = Config::varZ;
    }

    MPlexQF initPhi;
    MPlexQI xtou;  // bool to determine "split space", i.e. map x to u or v
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      initPhi.At(n, 0, 0) = std::abs(getPhi(x.constAt(n, 0, 0), y.constAt(n, 0, 0)));
      xtou.At(n, 0, 0) =
          ((initPhi.constAt(n, 0, 0) < Const::PIOver4 || initPhi.constAt(n, 0, 0) > Const::PI3Over4) ? 1 : 0);
    }

    MPlexHV u, v;
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      if (xtou.At(n, 0, 0))  // x mapped to u
      {
        for (int i = 0; i < 3; ++i) {
          u.At(n, i, 0) = x.constAt(n, i, 0) / r2.constAt(n, i, 0);
          v.At(n, i, 0) = y.constAt(n, i, 0) / r2.constAt(n, i, 0);
        }
      } else  // x mapped to v
      {
        for (int i = 0; i < 3; ++i) {
          u.At(n, i, 0) = y.constAt(n, i, 0) / r2.constAt(n, i, 0);
          v.At(n, i, 0) = x.constAt(n, i, 0) / r2.constAt(n, i, 0);
        }
      }
    }

    MPlexHH A;
    //#pragma omp simd // triggers an internal compiler error with icc 18.0.2!
    for (int n = 0; n < N; ++n) {
      for (int i = 0; i < 3; ++i) {
        A.At(n, i, 0) = 1.0f;
        A.At(n, i, 1) = -u.constAt(n, i, 0);
        A.At(n, i, 2) = -u.constAt(n, i, 0) * u.constAt(n, i, 0);
      }
    }
    Matriplex::invertCramer(A);
    MPlexHV C;
    CFMap(A, v, C);

    MPlexQF a, b;
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      b.At(n, 0, 0) = 1.0f / (2.0f * C.constAt(n, 0, 0));
      a.At(n, 0, 0) = b.constAt(n, 0, 0) * C.constAt(n, 1, 0);
    }

    // constant used throughtout
    const float k = (Const::sol * Config::Bfield) / 100.0f;

    MPlexQF vrx, vry, pT, px, py, pz;
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      vrx.At(n, 0, 0) =
          (xtou.constAt(n, 0, 0) ? x.constAt(n, 0, 0) - a.constAt(n, 0, 0) : x.constAt(n, 0, 0) - b.constAt(n, 0, 0));
      vry.At(n, 0, 0) =
          (xtou.constAt(n, 0, 0) ? y.constAt(n, 0, 0) - b.constAt(n, 0, 0) : y.constAt(n, 0, 0) - a.constAt(n, 0, 0));
      pT.At(n, 0, 0) = k * hipo(vrx.constAt(n, 0, 0), vry.constAt(n, 0, 0));
      px.At(n, 0, 0) = std::copysign(k * vry.constAt(n, 0, 0), x.constAt(n, 2, 0) - x.constAt(n, 0, 0));
      py.At(n, 0, 0) = std::copysign(k * vrx.constAt(n, 0, 0), y.constAt(n, 2, 0) - y.constAt(n, 0, 0));
      pz.At(n, 0, 0) = (pT.constAt(n, 0, 0) * (z.constAt(n, 2, 0) - z.constAt(n, 0, 0))) /
                       hipo((x.constAt(n, 2, 0) - x.constAt(n, 0, 0)), (y.constAt(n, 2, 0) - y.constAt(n, 0, 0)));
    }

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      outPar.At(n, 3, 0) = 1.0f / pT.constAt(n, 0, 0);
      outPar.At(n, 4, 0) = getPhi(px.constAt(n, 0, 0), py.constAt(n, 0, 0));
      outPar.At(n, 5, 0) = getTheta(pT.constAt(n, 0, 0), pz.constAt(n, 0, 0));
#ifdef INWARDFIT  // arctan is odd, so pz -> -pz means theta -> -theta
      if (fitting)
        outPar.At(n, 5, 0) *= -1.0f;
#endif
    }

#pragma omp simd
    for (int n = 0; n < N; ++n) {
      outErr.At(n, 3, 3) =
          (fitting ? Config::ptinverr049 * Config::ptinverr049 : Config::ptinverr012 * Config::ptinverr012);
      outErr.At(n, 4, 4) = (fitting ? Config::phierr049 * Config::phierr049 : Config::phierr012 * Config::phierr012);
      outErr.At(n, 5, 5) =
          (fitting ? Config::thetaerr049 * Config::thetaerr049 : Config::thetaerr012 * Config::thetaerr012);
    }

    if (debug && g_debug) {
      for (int n = 0; n < N; ++n) {
        dprintf("afterCF seedID: %1u \n", seedID.constAt(n, 0, 0));
        // do a dumb copy out
        TrackState updatedState;
        for (int i = 0; i < 6; i++) {
          updatedState.parameters[i] = outPar.constAt(n, i, 0);
          for (int j = 0; j < 6; j++) {
            updatedState.errors[i][j] = outErr.constAt(n, i, j);
          }
        }

        dcall(print("CCS", updatedState));
        updatedState.convertFromCCSToCartesian();
        dcall(print("Pol", updatedState));
        dprint("--------------------------------");
      }
    }
  }

}  // end namespace mkfit
