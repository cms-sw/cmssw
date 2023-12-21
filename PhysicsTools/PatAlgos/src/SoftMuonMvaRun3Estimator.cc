#include "PhysicsTools/PatAlgos/interface/SoftMuonMvaRun3Estimator.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "PhysicsTools/PatAlgos/interface/XGBooster.h"

typedef std::pair<const reco::MuonChamberMatch*, const reco::MuonSegmentMatch*> MatchPair;

const MatchPair& getBetterMatch(const MatchPair& match1, const MatchPair& match2) {
  // Prefer DT over CSC simply because it's closer to IP
  // and will have less multiple scattering (at least for
  // RB1 vs ME1/3 case). RB1 & ME1/2 overlap is tiny
  if (match2.first->detector() == MuonSubdetId::DT and match1.first->detector() != MuonSubdetId::DT)
    return match2;

  // For the rest compare local x match. We expect that
  // segments belong to the muon, so the difference in
  // local x is a reflection on how well we can measure it
  if (abs(match1.first->x - match1.second->x) > abs(match2.first->x - match2.second->x))
    return match2;

  return match1;
}

float dX(const MatchPair& match) {
  if (match.first and match.second->hasPhi())
    return (match.first->x - match.second->x);
  else
    return 9999.;
}

float pullX(const MatchPair& match) {
  if (match.first and match.second->hasPhi())
    return dX(match) / sqrt(pow(match.first->xErr, 2) + pow(match.second->xErr, 2));
  else
    return 9999.;
}

float pullDxDz(const MatchPair& match) {
  if (match.first and match.second->hasPhi())
    return (match.first->dXdZ - match.second->dXdZ) /
           sqrt(pow(match.first->dXdZErr, 2) + pow(match.second->dXdZErr, 2));
  else
    return 9999.;
}

float dY(const MatchPair& match) {
  if (match.first and match.second->hasZed())
    return (match.first->y - match.second->y);
  else
    return 9999.;
}

float pullY(const MatchPair& match) {
  if (match.first and match.second->hasZed())
    return dY(match) / sqrt(pow(match.first->yErr, 2) + pow(match.second->yErr, 2));
  else
    return 9999.;
}

float pullDyDz(const MatchPair& match) {
  if (match.first and match.second->hasZed())
    return (match.first->dYdZ - match.second->dYdZ) /
           sqrt(pow(match.first->dYdZErr, 2) + pow(match.second->dYdZErr, 2));
  else
    return 9999.;
}

void fillMatchInfoForStation(std::string prefix, pat::XGBooster& booster, const MatchPair& match) {
  booster.set(prefix + "_dX", dX(match));
  booster.set(prefix + "_pullX", pullX(match));
  booster.set(prefix + "_pullDxDz", pullDxDz(match));
  booster.set(prefix + "_dY", dY(match));
  booster.set(prefix + "_pullY", pullY(match));
  booster.set(prefix + "_pullDyDz", pullDyDz(match));
}

void fillMatchInfo(pat::XGBooster& booster, const pat::Muon& muon) {
  // Initiate containter for results
  const int n_stations = 2;
  std::vector<MatchPair> matches;
  for (unsigned int i = 0; i < n_stations; ++i)
    matches.push_back(std::pair(nullptr, nullptr));

  // Find best matches
  for (auto& chamberMatch : muon.matches()) {
    unsigned int station = chamberMatch.station() - 1;
    if (station >= n_stations)
      continue;

    // Find best segment match.
    // We could consider all segments, but we will restrict to segments
    // that match to this candidate better than to other muon candidates
    for (auto& segmentMatch : chamberMatch.segmentMatches) {
      if (not segmentMatch.isMask(reco::MuonSegmentMatch::BestInStationByDR) ||
          not segmentMatch.isMask(reco::MuonSegmentMatch::BelongsToTrackByDR))
        continue;

      // Multiple segment matches are possible in different
      // chambers that are either overlapping or belong to
      // different detectors. We need to select one.
      auto match_pair = MatchPair(&chamberMatch, &segmentMatch);

      if (matches[station].first)
        matches[station] = getBetterMatch(matches[station], match_pair);
      else
        matches[station] = match_pair;
    }
  }

  // Fill matching information
  fillMatchInfoForStation("match1", booster, matches[0]);
  fillMatchInfoForStation("match2", booster, matches[1]);
}

float pat::computeSoftMvaRun3(pat::XGBooster& booster, const pat::Muon& muon) {
  if (!muon.isTrackerMuon() && !muon.isGlobalMuon())
    return 0;

  fillMatchInfo(booster, muon);

  booster.set("pt", muon.pt());
  booster.set("eta", muon.eta());
  booster.set("trkValidFrac", muon.innerTrack()->validFraction());
  booster.set("glbTrackProbability", muon.combinedQuality().glbTrackProbability);
  booster.set("nLostHitsInner",
              muon.innerTrack()->hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS));
  booster.set("nLostHitsOuter",
              muon.innerTrack()->hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS));
  booster.set("trkKink", muon.combinedQuality().trkKink);
  booster.set("chi2LocalPosition", muon.combinedQuality().chi2LocalPosition);
  booster.set("nPixels", muon.innerTrack()->hitPattern().numberOfValidPixelHits());
  booster.set("nValidHits", muon.innerTrack()->hitPattern().numberOfValidTrackerHits());
  booster.set("nLostHitsOn", muon.innerTrack()->hitPattern().numberOfLostTrackerHits(reco::HitPattern::TRACK_HITS));
  booster.set("glbNormChi2", muon.isGlobalMuon() ? muon.globalTrack()->normalizedChi2() : 9999.);
  booster.set("trkLayers", muon.innerTrack()->hitPattern().trackerLayersWithMeasurement());
  booster.set("highPurity", muon.innerTrack()->quality(reco::Track::highPurity));

  return booster.predict();
}
