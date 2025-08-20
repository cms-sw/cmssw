#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "PropagationMPlex.h"

#include <vdt/log.h>
#include <vdt/sincos.h>

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  // this version does not assume to know which elements are 0 or 1, so it does the full multiplication
  void MultHelixPropFull(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
// optimization reports indicate only the inner two loops are good
// candidates for vectorization
#pragma omp simd
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
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
// optimization reports indicate only the inner two loops are good
// candidates for vectorization
#pragma omp simd
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += B.constAt(n, i, k) * A.constAt(n, j, k);
        }
      }
    }
  }

#ifdef UNUSED
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

  //==============================================================================

  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            const MPlexHV& plNrm,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            const int N_proc) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      if (n >= N_proc)
        continue;
      float radL = hitsRl.constAt(n, 0, 0);
      if (radL < 1e-13f)
        continue;  //ugly, please fixme
      const float theta = outPar.constAt(n, 5, 0);
      // const float pt = 1.f / outPar.constAt(n, 3, 0);  //fixme, make sure it is positive?
      const float ipt = outPar.constAt(n, 3, 0);
      const float pt = 1.f / ipt;  //fixme, make sure it is positive?
      const float ipt2 = ipt * ipt;
      float sT;
      float cT;
      vdt::fast_sincosf(theta, sT, cT);
      const float p = pt / sT;
      const float pz = p * cT;
      const float p2 = p * p;
      constexpr float mpi = 0.140;       // m=140 MeV, pion
      constexpr float mpi2 = mpi * mpi;  // m=140 MeV, pion
      const float beta2 = p2 / (p2 + mpi2);
      const float beta = std::sqrt(beta2);
      //radiation lenght, corrected for the crossing angle (cos alpha from dot product of radius vector and momentum)
      float sinP;
      float cosP;
      vdt::fast_sincosf(outPar.constAt(n, 4, 0), sinP, cosP);
      const float invCos = p / std::abs(pt * cosP * plNrm.constAt(n, 0, 0) + pt * sinP * plNrm.constAt(n, 1, 0) +
                                        pz * plNrm.constAt(n, 2, 0));
      radL = radL * invCos;  //fixme works only for barrel geom
      // multiple scattering
      //vary independently phi and theta by the rms of the planar multiple scattering angle
      // XXX-KMD radL < 0, see your fixme above! Repeating bailout
      if (radL < 1e-13f)
        continue;
      // const float thetaMSC = 0.0136f*std::sqrt(radL)*(1.f+0.038f*vdt::fast_logf(radL))/(beta*p);// eq 32.15
      // const float thetaMSC2 = thetaMSC*thetaMSC;
      const float thetaMSC = 0.0136f * (1.f + 0.038f * vdt::fast_logf(radL)) / (beta * p);  // eq 32.15
      const float thetaMSC2 = thetaMSC * thetaMSC * radL;
      if /*constexpr*/ (Config::usePtMultScat) {
        outErr.At(n, 3, 3) += thetaMSC2 * pz * pz * ipt2 * ipt2;
        outErr.At(n, 3, 5) -= thetaMSC2 * pz * ipt2;
        outErr.At(n, 4, 4) += thetaMSC2 * p2 * ipt2;
        outErr.At(n, 5, 5) += thetaMSC2;
      } else {
        outErr.At(n, 4, 4) += thetaMSC2;
        outErr.At(n, 5, 5) += thetaMSC2;
      }
      //std::cout << "beta=" << beta << " p=" << p << std::endl;
      //std::cout << "multiple scattering thetaMSC=" << thetaMSC << " thetaMSC2=" << thetaMSC2 << " radL=" << radL << std::endl;
      //std::cout << "radL=" << hitsRl.constAt(n, 0, 0) << " beta=" << beta << " invCos=" << invCos << " radLCorr=" << radL << " thetaMSC=" << thetaMSC << " thetaMSC2=" << thetaMSC2 << std::endl;
      // energy loss
      // XXX-KMD beta2 = 1 => 1 / sqrt(0)
      // const float gamma = 1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      // const float gamma2 = gamma*gamma;
      const float gamma2 = (p2 + mpi2) / mpi2;
      const float gamma = std::sqrt(gamma2);  //1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      constexpr float me = 0.0005;            // m=0.5 MeV, electron
      const float wmax = 2.f * me * beta2 * gamma2 / (1.f + 2.f * gamma * me / mpi + me * me / (mpi * mpi));
      constexpr float I = 16.0e-9 * 10.75;
      const float deltahalf =
          vdt::fast_logf(28.816e-9f * std::sqrt(2.33f * 0.498f) / I) + vdt::fast_logf(beta * gamma) - 0.5f;
      const float dEdx =
          beta < 1.f ? (2.f * (hitsXi.constAt(n, 0, 0) * invCos *
                               (0.5f * vdt::fast_logf(2.f * me * beta2 * gamma2 * wmax / (I * I)) - beta2 - deltahalf) /
                               beta2))
                     : 0.f;  //protect against infs and nans
      // dEdx = dEdx*2.;//xi in cmssw is defined with an extra factor 0.5 with respect to formula 27.1 in pdg
      //std::cout << "dEdx=" << dEdx << " delta=" << deltahalf << " wmax=" << wmax << " Xi=" << hitsXi.constAt(n,0,0) << std::endl;
      const float dP = propSign.constAt(n, 0, 0) * dEdx / beta;
      outPar.At(n, 3, 0) = p / (std::max(p - dP, 0.001f) * pt);  //stay above 1MeV
      //assume 100% uncertainty
      outErr.At(n, 3, 3) += dP * dP / (p2 * pt * pt);
    }
  }

}  // namespace mkfit
