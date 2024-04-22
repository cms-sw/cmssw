#include "RecoBTag/FeatureTools/interface/LostTracksConverter.h"

namespace btagbtvdeep {

  void packedCandidateToFeatures(const pat::PackedCandidate* c_pf,
                                 const pat::Jet& jet,
                                 const TrackInfoBuilder& track_info,
                                 const bool isWeightedJet,
                                 const float drminpfcandsv,
                                 const float jetR,
                                 const float puppiw,
                                 LostTracksFeatures& lt_features,
                                 const bool flip,
                                 const float distminpfcandsv) {
    commonCandidateToFeatures(
        c_pf, jet, track_info, isWeightedJet, drminpfcandsv, jetR, puppiw, lt_features, flip, distminpfcandsv);

    lt_features.puppiw = puppiw;
    lt_features.charge = c_pf->charge();

    lt_features.lostInnerHits = catch_infs(c_pf->lostInnerHits(), 2);
    lt_features.numberOfPixelHits = catch_infs(c_pf->numberOfPixelHits(), -1);
    lt_features.numberOfStripHits = catch_infs(c_pf->stripLayersWithMeasurement(), -1);

    // if PackedCandidate does not have TrackDetails this gives an Exception
    // because unpackCovariance might be called for pseudoTrack/bestTrack
    if (c_pf->hasTrackDetails()) {
      const auto& pseudo_track = c_pf->pseudoTrack();
      lt_features.chi2 = catch_infs_and_bound(pseudo_track.normalizedChi2(), 300, -1, 300);
      // this returns the quality enum not a mask.
      lt_features.quality = pseudo_track.qualityMask();
    } else {
      // default negative chi2 and loose track if notTrackDetails
      lt_features.chi2 = catch_infs_and_bound(-1, 300, -1, 300);
      lt_features.quality = (1 << reco::TrackBase::loose);
    }
  }
  void recoCandidateToFeatures(const reco::PFCandidate* c_pf,
                               const reco::Jet& jet,
                               const TrackInfoBuilder& track_info,
                               const bool isWeightedJet,
                               const float drminpfcandsv,
                               const float jetR,
                               const float puppiw,
                               const int pv_ass_quality,
                               const reco::VertexRef& pv,
                               LostTracksFeatures& lt_features,
                               const bool flip,
                               const float distminpfcandsv) {
    commonCandidateToFeatures(
        c_pf, jet, track_info, isWeightedJet, drminpfcandsv, jetR, puppiw, lt_features, flip, distminpfcandsv);

    lt_features.puppiw = puppiw;

    const auto& pseudo_track = (c_pf->bestTrack()) ? *c_pf->bestTrack() : reco::Track();
    lt_features.chi2 = catch_infs_and_bound(std::floor(pseudo_track.normalizedChi2()), 300, -1, 300);
    lt_features.quality = quality_from_pfcand(*c_pf);
  }

}  // namespace btagbtvdeep
