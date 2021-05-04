///=== This is the Kalman Combinatorial Filter for 4 helix parameters track fit algorithm.

#include "L1Trigger/TrackFindingTMTT/interface/KFParamsComb.h"
#include "L1Trigger/TrackFindingTMTT/interface/KalmanState.h"
#include "L1Trigger/TrackFindingTMTT/interface/PrintL1trk.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <array>
#include <sstream>

using namespace std;

namespace tmtt {

  /* Initialize */

  KFParamsComb::KFParamsComb(const Settings* settings, const uint nHelixPar, const std::string& fitterName)
      : KFbase(settings, nHelixPar, fitterName),
        // Initialize cuts applied to helix states vs KF layer number of last added stub.
        kfLayerVsPtToler_(settings->kfLayerVsPtToler()),
        kfLayerVsD0Cut5_(settings->kfLayerVsD0Cut5()),
        kfLayerVsZ0Cut5_(settings->kfLayerVsZ0Cut5()),
        kfLayerVsZ0Cut4_(settings->kfLayerVsZ0Cut4()),
        kfLayerVsChiSq5_(settings->kfLayerVsChiSq5()),
        kfLayerVsChiSq4_(settings->kfLayerVsChiSq4()) {}

  /* Helix state seed  */

  TVectorD KFParamsComb::seedX(const L1track3D& l1track3D) const {
    TVectorD vecX(nHelixPar_);
    vecX[INV2R] = settings_->invPtToInvR() * l1track3D.qOverPt() / 2;
    vecX[PHI0] = reco::deltaPhi(l1track3D.phi0() - sectorPhi(), 0.);
    vecX[Z0] = l1track3D.z0();
    vecX[T] = l1track3D.tanLambda();
    if (nHelixPar_ == 5) {  // fit without d0 constraint
      vecX[D0] = l1track3D.d0();
    }

    return vecX;
  }

  /* Helix state seed covariance matrix */

  TMatrixD KFParamsComb::seedC(const L1track3D& l1track3D) const {
    TMatrixD matC(nHelixPar_, nHelixPar_);

    double invPtToInv2R = settings_->invPtToInvR() / 2;

    // Assumed track seed (from HT) uncertainty in transverse impact parameter.

    // Constants optimised by hand for TMTT algo.
    const float inv2Rsigma = 0.0314 * invPtToInv2R;
    constexpr float phi0sigma = 0.0102;
    constexpr float z0sigma = 5.0;
    constexpr float tanLsigma = 0.5;
    constexpr float d0Sigma = 1.0;
    // (z0, tanL, d0) uncertainties could be smaller for Hybrid, if seeded in PS? -- To check!
    // if (L1track3D.seedPS() > 0) z0sigma /= 4; ???
    matC[INV2R][INV2R] = pow(inv2Rsigma, 2);
    matC[PHI0][PHI0] = pow(phi0sigma, 2);
    matC[Z0][Z0] = pow(z0sigma, 2);
    matC[T][T] = pow(tanLsigma, 2);
    if (nHelixPar_ == 5) {  // fit without d0 constraint
      matC[D0][D0] = pow(d0Sigma, 2);
    }
    return matC;
  }

  /* Stub position measurements in (phi,z) */

  TVectorD KFParamsComb::vectorM(const Stub* stub) const {
    TVectorD meas(2);
    meas[PHI] = reco::deltaPhi(stub->phi(), sectorPhi());
    meas[Z] = stub->z();
    return meas;
  }

  // Stub position resolution in (phi,z)

  TMatrixD KFParamsComb::matrixV(const Stub* stub, const KalmanState* state) const {
    // Take Pt from input track candidate as more stable.
    double inv2R = (settings_->invPtToInvR()) * 0.5 * state->candidate().qOverPt();
    double inv2R2 = inv2R * inv2R;

    double tanl = state->vectorX()(T);  // factor of 0.9 improves rejection
    double tanl2 = tanl * tanl;

    double vphi(0);
    double vz(0);
    double vcorr(0);

    double a = stub->sigmaPerp() * stub->sigmaPerp();
    double b = stub->sigmaPar() * stub->sigmaPar();
    double r2 = stub->r() * stub->r();
    double invr2 = 1. / r2;

    // Scattering term scaling as 1/Pt.
    double sigmaScat = settings_->kalmanMultiScattTerm() / (state->candidate().pt());
    double sigmaScat2 = sigmaScat * sigmaScat;

    if (stub->barrel()) {
      vphi = (a * invr2) + sigmaScat2;

      if (stub->tiltedBarrel()) {
        // Convert uncertainty in (r,phi) to (z,phi).
        float scaleTilted = 1.;
        if (settings_->kalmanHOtilted()) {
          if (settings_->useApproxB()) {  // Simple firmware approximation
            scaleTilted = approxB(stub->z(), stub->r());
          } else {  // Exact C++ implementation.
            float tilt = stub->tiltAngle();
            scaleTilted = sin(tilt) + cos(tilt) * tanl;
          }
        }
        float scaleTilted2 = scaleTilted * scaleTilted;
        // This neglects the non-radial strip effect, assumed negligeable for PS.
        vz = b * scaleTilted2;
      } else {
        vz = b;
      }

      if (settings_->kalmanHOfw()) {
        // Use approximation corresponding to current firmware.
        vz = b;
      }

    } else {
      vphi = a * invr2 + sigmaScat2;
      vz = (b * tanl2);

      if (not stub->psModule()) {  // Neglect these terms in PS
        double beta = 0.;
        // Add correlation term related to conversion of stub residuals from (r,phi) to (z,phi).
        if (settings_->kalmanHOprojZcorr() == 2)
          beta += -inv2R;
        // Add alpha correction for non-radial 2S endcap strips..
        if (settings_->kalmanHOalpha() == 2)
          beta += -stub->alpha();  // alpha is 0 except in endcap 2S disks

        double beta2 = beta * beta;
        vphi += b * beta2;
        vcorr = b * (beta * tanl);

        if (settings_->kalmanHOfw()) {
          // Use approximation corresponding to current firmware.
          vphi = (a * invr2) + (b * inv2R2) + sigmaScat2;
          vcorr = 0.;
          vz = (b * tanl2);
        }
      }
    }

    TMatrixD matV(2, 2);
    matV(PHI, PHI) = vphi;
    matV(Z, Z) = vz;
    matV(PHI, Z) = vcorr;
    matV(Z, PHI) = vcorr;

    return matV;
  }

  /* The Kalman measurement matrix = derivative of helix intercept w.r.t. helix params */
  /* Here I always measure phi(r), and z(r) */

  TMatrixD KFParamsComb::matrixH(const Stub* stub) const {
    TMatrixD matH(2, nHelixPar_);
    double r = stub->r();
    matH(PHI, INV2R) = -r;
    matH(PHI, PHI0) = 1;
    if (nHelixPar_ == 5) {  // fit without d0 constraint
      matH(PHI, D0) = -1. / r;
    }
    matH(Z, Z0) = 1;
    matH(Z, T) = r;
    return matH;
  }

  /* Kalman helix ref point extrapolation matrix */

  TMatrixD KFParamsComb::matrixF(const Stub* stub, const KalmanState* state) const {
    const TMatrixD unitMatrix(TMatrixD::kUnit, TMatrixD(nHelixPar_, nHelixPar_));
    return unitMatrix;
  }

  /* Get physical helix params */

  TVectorD KFParamsComb::trackParams(const KalmanState* state) const {
    TVectorD vecX = state->vectorX();
    TVectorD vecY(nHelixPar_);
    vecY[QOVERPT] = 2. * vecX[INV2R] / settings_->invPtToInvR();
    vecY[PHI0] = reco::deltaPhi(vecX[PHI0] + sectorPhi(), 0.);
    vecY[Z0] = vecX[Z0];
    vecY[T] = vecX[T];
    if (nHelixPar_ == 5) {  // fit without d0 constraint
      vecY[D0] = vecX[D0];
    }
    return vecY;
  }

  /* If using 5 param helix fit, get track params with beam-spot constraint & track fit chi2 from applying it. */
  /* (N.B. chi2rz unchanged by constraint) */

  TVectorD KFParamsComb::trackParams_BeamConstr(const KalmanState* state, double& chi2rphi) const {
    if (nHelixPar_ == 5) {  // fit without d0 constraint
      TVectorD vecX = state->vectorX();
      TMatrixD matC = state->matrixC();
      TVectorD vecY(nHelixPar_);
      double delChi2rphi = (vecX[D0] * vecX[D0]) / matC[D0][D0];
      chi2rphi = state->chi2rphi() + delChi2rphi;
      // Apply beam-spot constraint to helix params in transverse plane only, as most sensitive to it.
      vecX[INV2R] -= vecX[D0] * (matC[INV2R][D0] / matC[D0][D0]);
      vecX[PHI0] -= vecX[D0] * (matC[PHI0][D0] / matC[D0][D0]);
      vecX[D0] = 0.0;
      vecY[QOVERPT] = 2. * vecX[INV2R] / settings_->invPtToInvR();
      vecY[PHI0] = reco::deltaPhi(vecX[PHI0] + sectorPhi(), 0.);
      vecY[Z0] = vecX[Z0];
      vecY[T] = vecX[T];
      vecY[D0] = vecX[D0];
      return vecY;
    } else {
      return (this->trackParams(state));
    }
  }

  /* Check if helix state passes cuts */

  bool KFParamsComb::isGoodState(const KalmanState& state) const {
    // Set cut values that are different for 4 & 5 param helix fits.
    vector<double> kfLayerVsZ0Cut = (nHelixPar_ == 5) ? kfLayerVsZ0Cut5_ : kfLayerVsZ0Cut4_;
    vector<double> kfLayerVsChiSqCut = (nHelixPar_ == 5) ? kfLayerVsChiSq5_ : kfLayerVsChiSq4_;

    unsigned nStubLayers = state.nStubLayers();
    bool goodState(true);

    TVectorD vecY = trackParams(&state);
    double qOverPt = vecY[QOVERPT];
    double pt = std::abs(1 / qOverPt);
    double z0 = std::abs(vecY[Z0]);

    // state parameter selections

    if (z0 > kfLayerVsZ0Cut[nStubLayers])
      goodState = false;
    if (pt < settings_->houghMinPt() - kfLayerVsPtToler_[nStubLayers])
      goodState = false;
    if (nHelixPar_ == 5) {  // fit without d0 constraint
      double d0 = std::abs(state.vectorX()[D0]);
      if (d0 > kfLayerVsD0Cut5_[nStubLayers])
        goodState = false;
    }

    // chi2 selection

    double chi2scaled = state.chi2scaled();  // chi2(r-phi) scaled down to improve electron performance.

    if (chi2scaled > kfLayerVsChiSqCut[nStubLayers])
      goodState = false;  // No separate pT selection needed

    const bool countUpdateCalls = false;  // Print statement to count calls to Updator.

    if (countUpdateCalls || (settings_->kalmanDebugLevel() >= 2 && tpa_ != nullptr) ||
        (settings_->kalmanDebugLevel() >= 2 && settings_->hybrid())) {
      std::stringstream text;
      text << std::fixed << std::setprecision(4);
      if (not goodState)
        text << "State veto:";
      if (goodState)
        text << "State kept:";
      text << " nlay=" << nStubLayers << " nskip=" << state.nSkippedLayers() << " chi2_scaled=" << chi2scaled;
      if (tpa_ != nullptr)
        text << " pt(mc)=" << tpa_->pt();
      text << " pt=" << pt << " q/pt=" << qOverPt << " tanL=" << vecY[T] << " z0=" << vecY[Z0]
           << " phi0=" << vecY[PHI0];
      if (nHelixPar_ == 5)  // fit without d0 constraint
        text << " d0=" << vecY[D0];
      text << " fake" << (tpa_ == nullptr);
      if (tpa_ != nullptr)
        text << " pt(mc)=" << tpa_->pt();
      PrintL1trk() << text.str();
    }

    return goodState;
  }

}  // namespace tmtt
