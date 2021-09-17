#ifndef L1Trigger_TrackFindingTMTT_KFParamsComb_h
#define L1Trigger_TrackFindingTMTT_KFParamsComb_h

#include "L1Trigger/TrackFindingTMTT/interface/KFbase.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"

///=== This is the Kalman Combinatorial Filter for 4 & 5 helix parameters track fit algorithm.
///=== All variable names & equations come from Fruhwirth KF paper
///=== http://dx.doi.org/10.1016/0168-9002%2887%2990887-4

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
