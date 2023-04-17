#ifndef RecoTracker_MkFitCore_src_MkBase_h
#define RecoTracker_MkFitCore_src_MkBase_h

#include "Matrix.h"

#include "PropagationMPlex.h"

namespace mkfit {

  class PropagationFlags;

  //==============================================================================
  // MkBase
  //==============================================================================

  class MkBase {
  public:
    static constexpr int iC = 0;  // current
    static constexpr int iP = 1;  // propagated

    float getPar(int itrack, int i, int par) const { return m_Par[i].constAt(itrack, par, 0); }

    float radiusSqr(int itrack, int i) const { return hipo_sqr(getPar(itrack, i, 0), getPar(itrack, i, 1)); }

    //----------------------------------------------------------------------------

    MkBase() {}

    //----------------------------------------------------------------------------

    void propagateTracksToR(float r, const int N_proc, const PropagationFlags &pf) {
      MPlexQF msRad;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msRad.At(n, 0, 0) = r;
      }

      propagateHelixToRMPlex(m_Err[iC], m_Par[iC], m_Chg, msRad, m_Err[iP], m_Par[iP], m_FailFlag, N_proc, pf);
    }

    void propagateTracksToHitR(const MPlexHV& par,
                               const int N_proc,
                               const PropagationFlags &pf,
                               const MPlexQI* noMatEffPtr = nullptr) {
      MPlexQF msRad;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msRad.At(n, 0, 0) = std::hypot(par.constAt(n, 0, 0), par.constAt(n, 1, 0));
      }

      propagateHelixToRMPlex(
          m_Err[iC], m_Par[iC], m_Chg, msRad, m_Err[iP], m_Par[iP], m_FailFlag, N_proc, pf, noMatEffPtr);
    }

    //----------------------------------------------------------------------------

    void propagateTracksToZ(float z, const int N_proc, const PropagationFlags &pf) {
      MPlexQF msZ;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msZ.At(n, 0, 0) = z;
      }

      propagateHelixToZMPlex(m_Err[iC], m_Par[iC], m_Chg, msZ, m_Err[iP], m_Par[iP], m_FailFlag, N_proc, pf);
    }

    void propagateTracksToHitZ(const MPlexHV& par,
                               const int N_proc,
                               const PropagationFlags &pf,
                               const MPlexQI* noMatEffPtr = nullptr) {
      MPlexQF msZ;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        msZ.At(n, 0, 0) = par.constAt(n, 2, 0);
      }

      propagateHelixToZMPlex(
          m_Err[iC], m_Par[iC], m_Chg, msZ, m_Err[iP], m_Par[iP], m_FailFlag, N_proc, pf, noMatEffPtr);
    }

    void propagateTracksToPCAZ(const int N_proc, const PropagationFlags &pf) {
      MPlexQF msZ;  // PCA z-coordinate
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
        const float slope = std::tan(m_Par[iC].constAt(n, 5, 0));
        //      msZ.At(n, 0, 0) = ( Config::beamspotz0 + slope * ( Config::beamspotr0 - std::hypot(m_Par[iC].constAt(n, 0, 0), m_Par[iC].constAt(n, 1, 0))) + slope * slope * m_Par[iC].constAt(n, 2, 0) ) / ( 1+slope*slope); // PCA w.r.t. z0, r0
        msZ.At(n, 0, 0) = (slope * (slope * m_Par[iC].constAt(n, 2, 0) -
                                    std::hypot(m_Par[iC].constAt(n, 0, 0), m_Par[iC].constAt(n, 1, 0)))) /
                          (1 + slope * slope);  // PCA to origin
      }

      propagateHelixToZMPlex(m_Err[iC], m_Par[iC], m_Chg, msZ, m_Err[iP], m_Par[iP], m_FailFlag, N_proc, pf);
    }

    void clearFailFlag() { m_FailFlag.setVal(0); }

    //----------------------------------------------------------------------------

  protected:
    MPlexLS m_Err[2];
    MPlexLV m_Par[2];
    MPlexQI m_Chg;
    MPlexQI m_FailFlag;
  };

}  // end namespace mkfit
#endif
