#ifndef DataFormats_TrackSoA_PixelTrackSoATab_h
#define DataFormats_TrackSoA_PixelTrackSoATab_h

#include <Eigen/Core>
#include <cmath>

#include "DataFormats/TrackSoA/interface/TracksSoA.h"

class PixelTrackSoATab {
  // Indices to the 5-dimensional track state vector (CMS convention: {phi, tip, q/pT, cotTheta, zip})
  static constexpr auto kStatePhi = 0;
  static constexpr auto kStateDxy = 1;
  static constexpr auto kStateQOverPt = 2;  // signed q/pT (carries the charge sign)
  static constexpr auto kStateDz = 4;

  // Indices into the 5x5 track covariance matrix (CMS convention)
  static constexpr auto kCovPhiPhi = 0;             // (0,0)
  static constexpr auto kCovPhiDxy = 1;             // (0,1)
  static constexpr auto kCovPhiQOverPt = 2;         // (0,2)
  static constexpr auto kCovDxyDxy = 5;             // (1,1)
  static constexpr auto kCovDxyQOverPt = 6;         // (1,2)
  static constexpr auto kCovQOverPtQOverPt = 9;     // (2,2)
  static constexpr auto kCovCotThetaCotTheta = 12;  // (3,3)
  static constexpr auto kCovCotThetaDz = 13;        // (3,4)
  static constexpr auto kCovDzDz = 14;              // (4,4)

public:
  PixelTrackSoATab() = default;

  PixelTrackSoATab(const reco::TrackSoAConstView& tracks, int idx) {
    const auto& track = tracks[idx];
    const auto& cov = track.covariance();
    const auto& state = track.state();
    const auto numHits = reco::nHits(tracks, idx);

    chi2_ = track.chi2();  // in the SoA chi2 is stored as chi2/ndof
    dzError_ = sqrt(cov(kCovDzDz));
    dxyError_ = sqrt(cov(kCovDxyDxy));
    eta_ = track.eta();
    nHits_ = numHits;
    phi_ = state(kStatePhi);
    phiError_ = sqrt(cov(kCovPhiPhi));
    pt_ = track.pt();
    qOverPtError_ = sqrt(cov(kCovQOverPtQOverPt));
    qOverPt_ = state(kStateQOverPt);      // signed q/pT (state(2))
    charge_ = reco::charge(tracks, idx);  // charge = sign(state(2))
    dzBS_ = state(kStateDz);
    dxyBS_ = state(kStateDxy);
    nLayers_ = track.nLayers();
    cotThetaError_ = sqrt(cov(kCovCotThetaCotTheta));
    covCotThetaDz_ = cov(kCovCotThetaDz);
    covDxyQOverPt_ = cov(kCovDxyQOverPt);
    covPhiDxy_ = cov(kCovPhiDxy);
    covPhiQOverPt_ = cov(kCovPhiQOverPt);
  }

  float chi2() const { return chi2_; }
  float dzError() const { return dzError_; }
  float dxyError() const { return dxyError_; }
  float eta() const { return eta_; }
  float nHits() const { return nHits_; }
  float phi() const { return phi_; }
  float phiError() const { return phiError_; }
  float pt() const { return pt_; }
  float qOverPtError() const { return qOverPtError_; }
  float qOverPt() const { return qOverPt_; }  // signed q/pT
  float charge() const { return charge_; }    // charge sign (+1/-1/0)
  float dzBS() const { return dzBS_; }
  float dxyBS() const { return dxyBS_; }
  float nLayers() const { return nLayers_; }
  float cotThetaError() const { return cotThetaError_; }
  float covCotThetaDz() const { return covCotThetaDz_; }
  float covDxyQOverPt() const { return covDxyQOverPt_; }
  float covPhiDxy() const { return covPhiDxy_; }
  float covPhiQOverPt() const { return covPhiQOverPt_; }

private:
  float chi2_, dzError_, dxyError_, eta_, nHits_, phi_, phiError_, pt_;
  float qOverPtError_, qOverPt_, charge_, dzBS_, dxyBS_, nLayers_, cotThetaError_;
  float covCotThetaDz_, covDxyQOverPt_, covPhiDxy_, covPhiQOverPt_;
};
#endif
