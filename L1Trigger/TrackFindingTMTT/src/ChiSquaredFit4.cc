#include "L1Trigger/TrackFindingTMTT/interface/ChiSquaredFit4.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;

namespace tmtt {

  ChiSquaredFit4::ChiSquaredFit4(const Settings* settings, const uint nPar) : ChiSquaredFitBase(settings, nPar) {
    largestresid_ = -1.0;
    ilargestresid_ = -1;
  }

  TVectorD ChiSquaredFit4::seed(const L1track3D& l1track3D) {
    TVectorD x(4);
    x[INVR] = settings_->invPtToInvR() * l1track3D.qOverPt();
    x[PHI0] = l1track3D.phi0();
    x[T] = l1track3D.tanLambda();
    x[Z0] = l1track3D.z0();
    return x;
  }

  //=== Calculate derivatives of track intercept with respect to track parameters

  TMatrixD ChiSquaredFit4::D(const TVectorD& x) {
    TMatrixD D(2 * stubs_.size(), nPar_);  // Empty matrix
    D.Zero();
    int j = 0;
    double rInv = x[INVR];
    double phi0 = x[PHI0];
    double t = x[T];
    for (unsigned i = 0; i < stubs_.size(); i++) {
      double ri = stubs_[i]->r();
      if (stubs_[i]->barrel()) {
        // Derivatives of r*phi
        D(j, INVR) = -0.5 * ri * ri;
        D(j, PHI0) = ri;
        j++;
        // Derivatives of z
        D(j, T) = ri;
        D(j, Z0) = 1;
        j++;
      } else {
        double phii = stubs_[i]->phi();
        int iphi = stubs_[i]->iphi();

        // N.B. These represent HALF the width and number of strips of sensor.
        double width = 0.5 * stubs_[i]->trackerModule()->sensorWidth();
        double nstrip = 0.5 * stubs_[i]->nStrips();

        double Deltai = width * (iphi - nstrip) / nstrip;  // Non-radial endcap 2S strip correction
        if (stubs_[i]->z() > 0.0)
          Deltai = -Deltai;
        double DeltaiOverRi = Deltai / ri;
        double theta0 = DeltaiOverRi + (2. / 3.) * DeltaiOverRi * DeltaiOverRi * DeltaiOverRi;

        double phi_track = phi0 - 0.5 * rInv * ri;  //Expected phi hit given the track

        double tInv = 1 / t;
        // Derivatives of r
        D(j, INVR) = -1 * ri * ri * ri * rInv;
        D(j, PHI0) = 0;
        D(j, T) = -ri * tInv;
        D(j, Z0) = -1 * tInv;
        j++;
        // Derivatives of r*phi
        D(j, INVR) = -0.5 * ri * ri;
        D(j, PHI0) = ri;
        D(j, T) = ri * 0.5 * rInv * ri * tInv - ((phi_track - phii) - theta0) * ri * tInv;
        D(j, Z0) = ri * 0.5 * rInv * tInv - ((phi_track - phii) - theta0) * tInv;
        j++;
      }
    }
    return D;
  }

  //=== In principle, this is the stub position covariance matrix.
  //=== In practice, it misses a factor "sigma", because unconventionally, this is absorbed into residuals() function.

  TMatrixD ChiSquaredFit4::Vinv() {
    TMatrixD Vinv(2 * stubs_.size(), 2 * stubs_.size());
    // Scattering term scaling as 1/Pt.
    double sigmaScat = settings_->kalmanMultiScattTerm() * std::abs(qOverPt_seed_);
    for (unsigned i = 0; i < stubs_.size(); i++) {
      double sigmaPerp = stubs_[i]->sigmaPerp();
      double sigmaPar = stubs_[i]->sigmaPar();
      double ri = stubs_[i]->r();
      sigmaPerp = sqrt(sigmaPerp * sigmaPerp + sigmaScat * sigmaScat * ri * ri);
      if (stubs_[i]->barrel()) {
        Vinv(2 * i, 2 * i) = 1 / sigmaPerp;
        Vinv(2 * i + 1, 2 * i + 1) = 1 / sigmaPar;
      } else {
        Vinv(2 * i, 2 * i) = 1 / sigmaPar;
        Vinv(2 * i + 1, 2 * i + 1) = 1 / sigmaPerp;
      }
    }
    return Vinv;
  }

  //=== Calculate residuals w.r.t. track divided by uncertainty.

  TVectorD ChiSquaredFit4::residuals(const TVectorD& x) {
    unsigned int n = stubs_.size();

    TVectorD delta(2 * n);

    double rInv = x[INVR];
    double phi0 = x[PHI0];
    double t = x[T];
    double z0 = x[Z0];

    unsigned int j = 0;

    largestresid_ = -1.0;
    ilargestresid_ = -1;

    // Scattering term scaling as 1/Pt.
    double sigmaScat = settings_->kalmanMultiScattTerm() * std::abs(qOverPt_seed_);

    for (unsigned int i = 0; i < n; i++) {
      double ri = stubs_[i]->r();
      double zi = stubs_[i]->z();
      double phii = stubs_[i]->phi();
      double sigmaPerp = stubs_[i]->sigmaPerp();
      double sigmaPar = stubs_[i]->sigmaPar();
      sigmaPerp = sqrt(sigmaPerp * sigmaPerp + sigmaScat * sigmaScat * ri * ri);

      if (stubs_[i]->barrel()) {
        double halfRinvRi = 0.5 * ri * rInv;
        double aSinHalfRinvRi = halfRinvRi + (2. / 3.) * halfRinvRi * halfRinvRi * halfRinvRi;
        double deltaphi = reco::deltaPhi(phi0 - aSinHalfRinvRi - phii, 0.);
        delta[j++] = (ri * deltaphi) / sigmaPerp;
        delta[j++] = (z0 + (2.0 / rInv) * t * aSinHalfRinvRi - zi) / sigmaPar;
      } else {
        double tInv = 1 / t;
        double r_track = (zi - z0) * tInv;
        double phi_track = phi0 - 0.5 * rInv * (zi - z0) * tInv;
        int iphi = stubs_[i]->iphi();

        // N.B. These represent HALF the width and number of strips of sensor.
        double width = 0.5 * stubs_[i]->trackerModule()->sensorWidth();
        double nstrip = 0.5 * stubs_[i]->nStrips();

        double Deltai = width * (iphi - nstrip) / nstrip;  // Non-radial endcap 2S strip correction

        if (stubs_[i]->z() > 0.0)
          Deltai = -Deltai;

        double DeltaiOverRi = Deltai / ri;
        double theta0 = DeltaiOverRi + (2. / 3.) * DeltaiOverRi * DeltaiOverRi * DeltaiOverRi;
        double Delta = Deltai - r_track * (theta0 - (phi_track - phii));

        delta[j++] = (r_track - ri) / sigmaPar;
        delta[j++] = Delta / sigmaPerp;
      }

      if (std::abs(delta[j - 2]) > largestresid_) {
        largestresid_ = std::abs(delta[j - 2]);
        ilargestresid_ = i;
      }

      if (std::abs(delta[j - 1]) > largestresid_) {
        largestresid_ = std::abs(delta[j - 1]);
        ilargestresid_ = i;
      }
    }

    return delta;
  }

}  // namespace tmtt
