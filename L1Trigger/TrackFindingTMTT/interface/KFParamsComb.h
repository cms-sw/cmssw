#ifndef L1Trigger_TrackFindingTMTT_KFParamsComb_h
#define L1Trigger_TrackFindingTMTT_KFParamsComb_h

#include "L1Trigger/TrackFindingTMTT/interface/KFbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"

///=== This is the Kalman Combinatorial Filter for 4 & 5 helix parameters track fit algorithm.
///===
///=== All variable names & equations come from Fruhwirth KF paper
///=== http://dx.doi.org/10.1016/0168-9002%2887%2990887-4
///===
///=== Summary of variables:
///=== m = hit position (phi,z)
///=== V = hit position 2x2 covariance matrix in (phi,z).
///=== x = helix params
///=== C = helix params 4x4 covariance matrix
///=== r = residuals
///=== H = 2x4 derivative matrix (expected stub position w.r.t. helix params)
///=== K = KF gain 2x2 matrix
///=== x' & C': Updated values of x & C after KF iteration
///=== Boring: F = unit matrix; pxcov = C
///===
///=== Summary of equations:
///=== S = H*C (2x4 matrix); St = Transpose S
///=== R = V + H*C*Ht (KF paper) = V + H*St (used here at simpler): 2x2 matrix
///=== Rinv = Inverse R
///=== K = St * Rinv : 2x2 Kalman gain matrix * det(R)
///=== r = m - H*x
///=== x' = x + K*r
///=== C' = C - K*H*C (KF paper) = C - K*S (used here as simpler)
///=== delta(chi2) = r(transpose) * Rinv * r : Increase in chi2 from new stub added during iteration.

namespace tmtt {

  class KFParamsComb : public KFbase {
  public:
    KFParamsComb(const Settings* settings, const uint nHelixPar, const std::string& fitterName);

    ~KFParamsComb() override = default;

  protected:
    //--- Input data

    // Seed track helix params & covariance matrix
    TVectorD seedX(const L1track3D& l1track3D) const override;
    TMatrixD seedC(const L1track3D& l1track3D) const override;

    // Stub coordinate measurements & resolution
    TVectorD vectorM(const Stub* stub) const override;
    TMatrixD matrixV(const Stub* stub, const KalmanState* state) const override;

    //--- KF maths matrix multiplications

    // Derivate of helix intercept point w.r.t. helix params.
    TMatrixD matrixH(const Stub* stub) const override;
    // Kalman helix ref point extrapolation matrix
    TMatrixD matrixF(const Stub* stub, const KalmanState* state) const override;

    // Convert to physical helix params instead of local ones used by KF
    TVectorD trackParams(const KalmanState* state) const override;
    TVectorD trackParams_BeamConstr(const KalmanState* state, double& chi2rphi) const override;

    // Does helix state pass cuts?
    bool isGoodState(const KalmanState& state) const override;

  protected:
    std::vector<double> kfLayerVsPtToler_;
    std::vector<double> kfLayerVsD0Cut5_;
    std::vector<double> kfLayerVsZ0Cut5_;
    std::vector<double> kfLayerVsZ0Cut4_;
    std::vector<double> kfLayerVsChiSq5_;
    std::vector<double> kfLayerVsChiSq4_;
  };

}  // namespace tmtt

#endif
