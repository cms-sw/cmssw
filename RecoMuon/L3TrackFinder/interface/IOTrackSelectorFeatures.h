#ifndef RecoMuon_L3TrackFinder_IOTrackSelectorFeatures_h
#define RecoMuon_L3TrackFinder_IOTrackSelectorFeatures_h

// Features of the high-purity selection of inside-out muon tracks (pixel tracks and LST-seeded
// tracks), used by MuonIOTracksForestSelector.
//
// There are 46 features: 9 track quantities, the number of endcap stubs of the matched L1 tracker
// muon, and 36 matching quantities of the candidates in the muon stations (MuonChamberMatch.h).
// The struct gives each feature a name. toArray() gives the features in the order of the forest.
//
// Numeric rules (the training extraction must use the same rules): each raw input is rounded to
// float, all arithmetic is in double precision, and each feature is rounded to float once. The
// forest splits on exact float values, and some constant features are exactly on a split value.
// Thus a float or double difference between training and inference can change the score.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoMuon/L3TrackFinder/interface/MuonChamberMatch.h"

namespace muonhp {

  // Matching constants. They must be equal to the values of the training extraction.
  inline constexpr double kEpsilon = 1e-6;           // floor for logarithms and divisions
  inline constexpr double kMatchDR2Cut = 0.3 * 0.3;  // the training extraction also computes 0.3 * 0.3
  inline constexpr double kMatchChi2PtCut = 9.0;     // 3 sigma
  inline constexpr double kChi2PtEps = 1e-12;

  // Raw input rounded to float, as in the training input, and given back as double.
  inline double asStored(double x) { return static_cast<float>(x); }
  // log10(|x| + eps): the logarithmic compression of the features.
  inline double logFloor(double x) { return std::log10(std::abs(x) + kEpsilon); }

  // Track parameters for the chamber matching.
  inline ChamberMatchTrack chamberMatchTrack(const reco::Track& track) {
    const double qoverp = asStored(track.qoverp());
    return {asStored(track.pt()),
            asStored(track.eta()),
            asStored(track.phi()),
            static_cast<double>((qoverp > 0.) - (qoverp < 0.))};  // sign of q/p: -1, 0 or +1
  }

  // Number of endcap stubs of the L1 tracker muon matched to the track. The match is the L1 tracker muon nearest
  // in dR among those with a pT compatible with the track pT (chi2 < kMatchChi2PtCut), if dR^2 < kMatchDR2Cut.
  // The result is 0 if there is no match.
  inline double matchedL1EndcapStubs(
      double pt, double ptErr, double eta, double phi, const l1t::TrackerMuonCollection& l1TkMuons) {
    double minDR2 = std::numeric_limits<double>::max();
    int bestIndex = -1;
    for (size_t l1Idx = 0; l1Idx != l1TkMuons.size(); ++l1Idx) {
      const auto& l1TkMu = l1TkMuons[l1Idx];
      // pT, eta and phi of the L1 tracker muon (the values of its L1 track at the vertex)
      const double ptDiff = pt - asStored(l1TkMu.phPt());
      const double chi2Pt = (ptDiff * ptDiff) / (ptErr * ptErr + kChi2PtEps);
      if (chi2Pt >= kMatchChi2PtCut) {
        continue;
      }
      const double dR2 = reco::deltaR2(eta, phi, asStored(l1TkMu.phEta()), asStored(l1TkMu.phPhi()));
      if (dR2 < minDR2) {
        minDR2 = dR2;
        bestIndex = static_cast<int>(l1Idx);
      }
    }
    if (bestIndex < 0 || !(minDR2 < kMatchDR2Cut)) {
      return 0.;
    }
    int nEndcap = 0;
    for (const auto& stubRef : l1TkMuons[bestIndex].stubs()) {
      if (stubRef.isNonnull() && stubRef->type() == 0) {
        ++nEndcap;
      }
    }
    return nEndcap;
  }

  // The 46 features, in the order of toArray().
  struct IOTrackFeatures {
    // Track
    float logP = 0.f;
    float logPt = 0.f;
    float logDzErr = 0.f;
    float logQoverpErr = 0.f;
    float logSigmaPtOverPt = 0.f;
    float logSip2D = 0.f;
    float absEta = 0.f;
    float logEtaErr = 0.f;
    float logPhiErr = 0.f;
    // Matched L1 tracker muon
    float nStubsEndcap = 0.f;
    // Muon stations: the two candidates nearest in eta in each station (bend, dEta and, for segments, bendDiff;
    // see MuonChamberMatch.h)
    ChamberMatch chambers;

    static constexpr size_t kSize = 46;

    // Gives the features in the order of the forest. Do not change this order without a new training.
    std::array<float, kSize> toArray() const {
      const auto& c = chambers;
      return {// 0-8 track
              logP,
              logPt,
              logDzErr,
              logQoverpErr,
              logSigmaPtOverPt,
              logSip2D,
              absEta,
              logEtaErr,
              logPhiErr,
              // 9 L1 tracker muon
              nStubsEndcap,
              // 10-20 bend of the nearest candidate of each station
              c[kME0].nearest.bend,
              c[kSt1].nearest.bend,
              c[kSt2].nearest.bend,
              c[kSt3].nearest.bend,
              c[kSt4].nearest.bend,
              c[kGE11].nearest.bend,
              c[kGE21].nearest.bend,
              c[kRPC1].nearest.bend,
              c[kRPC2].nearest.bend,
              c[kRPC3].nearest.bend,
              c[kRPC4].nearest.bend,
              // 21-31 eta difference of the nearest candidate of each station
              c[kME0].nearest.dEta,
              c[kSt1].nearest.dEta,
              c[kSt2].nearest.dEta,
              c[kSt3].nearest.dEta,
              c[kSt4].nearest.dEta,
              c[kGE11].nearest.dEta,
              c[kGE21].nearest.dEta,
              c[kRPC1].nearest.dEta,
              c[kRPC2].nearest.dEta,
              c[kRPC3].nearest.dEta,
              c[kRPC4].nearest.dEta,
              // 32-36 bendDiff of the nearest candidate (segment stations)
              c[kME0].nearest.bendDiff,
              c[kSt1].nearest.bendDiff,
              c[kSt2].nearest.bendDiff,
              c[kSt3].nearest.bendDiff,
              c[kSt4].nearest.bendDiff,
              // 37-45 second candidate of the inner stations
              c[kME0].second.bend,
              c[kSt1].second.bend,
              c[kGE11].second.bend,
              c[kGE21].second.bend,
              c[kME0].second.dEta,
              c[kSt1].second.dEta,
              c[kGE11].second.dEta,
              c[kGE21].second.dEta,
              c[kSt1].second.bendDiff};
    }
  };

  inline IOTrackFeatures extractIOTrackFeatures(const reco::Track& track,
                                                const l1t::TrackerMuonCollection& l1TkMuons,
                                                const std::vector<ChamberPoint>& chamberPoints) {
    IOTrackFeatures f;

    // Raw track quantities
    const double p = asStored(track.p());
    const double pt = asStored(track.pt());
    const double ptErr = asStored(track.ptError());
    const double eta = asStored(track.eta());
    const double phi = asStored(track.phi());
    const double dxy = asStored(track.dxy());
    const double dxyErr = asStored(track.dxyError());

    f.logP = logFloor(p);
    f.logPt = logFloor(pt);
    f.logDzErr = logFloor(asStored(track.dzError()));
    f.logQoverpErr = logFloor(asStored(track.qoverpError()));
    f.logSigmaPtOverPt = std::log10(ptErr / std::max(pt, kEpsilon) + kEpsilon);
    f.logSip2D = std::log10(std::abs(dxy) / std::max(dxyErr, kEpsilon) + kEpsilon);
    f.absEta = std::abs(eta);
    f.logEtaErr = logFloor(asStored(track.etaError()));
    f.logPhiErr = logFloor(asStored(track.phiError()));

    f.nStubsEndcap = matchedL1EndcapStubs(pt, ptErr, eta, phi, l1TkMuons);

    f.chambers = matchChambers(chamberMatchTrack(track), chamberPoints);

    return f;
  }

}  // namespace muonhp

#endif
