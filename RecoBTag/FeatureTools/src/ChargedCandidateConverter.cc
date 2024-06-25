#include "RecoBTag/FeatureTools/interface/ChargedCandidateConverter.h"

namespace btagbtvdeep {

  void packedCandidateToFeatures(const pat::PackedCandidate* c_pf,
                                 const pat::Jet& jet,
                                 const TrackInfoBuilder& track_info,
                                 const bool isWeightedJet,
                                 const float drminpfcandsv,
                                 const float jetR,
                                 const float puppiw,
                                 ChargedCandidateFeatures& c_pf_features,
                                 const bool flip,
                                 const float distminpfcandsv) {
    commonCandidateToFeatures(
        c_pf, jet, track_info, isWeightedJet, drminpfcandsv, jetR, puppiw, c_pf_features, flip, distminpfcandsv);

    c_pf_features.vtx_ass = c_pf->pvAssociationQuality();

    c_pf_features.puppiw = puppiw;
    c_pf_features.charge = c_pf->charge();

    c_pf_features.CaloFrac = c_pf->caloFraction();
    c_pf_features.HadFrac = c_pf->hcalFraction();
    c_pf_features.lostInnerHits = catch_infs(c_pf->lostInnerHits(), 2);
    c_pf_features.numberOfPixelHits = catch_infs(c_pf->numberOfPixelHits(), -1);
    c_pf_features.numberOfStripHits = catch_infs(c_pf->stripLayersWithMeasurement(), -1);

    // if PackedCandidate does not have TrackDetails this gives an Exception
    // because unpackCovariance might be called for pseudoTrack/bestTrack
    if (c_pf->hasTrackDetails()) {
      const auto& pseudo_track = c_pf->pseudoTrack();
      c_pf_features.chi2 = catch_infs_and_bound(pseudo_track.normalizedChi2(), 300, -1, 300);
      // this returns the quality enum not a mask.
      c_pf_features.quality = pseudo_track.qualityMask();
    } else {
      // default negative chi2 and loose track if notTrackDetails
      c_pf_features.chi2 = catch_infs_and_bound(-1, 300, -1, 300);
      c_pf_features.quality = (1 << reco::TrackBase::loose);
    }

    c_pf_features.dxy = catch_infs(c_pf->dxy());
    c_pf_features.dz = catch_infs(c_pf->dz());
    c_pf_features.dxysig = c_pf->bestTrack() ? catch_infs(c_pf->dxy() / c_pf->dxyError()) : 0;
    c_pf_features.dzsig = c_pf->bestTrack() ? catch_infs(c_pf->dz() / c_pf->dzError()) : 0;

    float pdgid_;
    if (abs(c_pf->pdgId()) == 11 and c_pf->charge() != 0) {
      pdgid_ = 0.0;
    } else if (abs(c_pf->pdgId()) == 13 and c_pf->charge() != 0) {
      pdgid_ = 1.0;
    } else if (abs(c_pf->pdgId()) == 22 and c_pf->charge() == 0) {
      pdgid_ = 2.0;
    } else if (abs(c_pf->pdgId()) != 22 and c_pf->charge() == 0 and abs(c_pf->pdgId()) != 1 and
               abs(c_pf->pdgId()) != 2) {
      pdgid_ = 3.0;
    } else if (abs(c_pf->pdgId()) != 11 and abs(c_pf->pdgId()) != 13 and c_pf->charge() != 0) {
      pdgid_ = 4.0;
    } else if (c_pf->charge() == 0 and abs(c_pf->pdgId()) == 1) {
      pdgid_ = 5.0;
    } else if (c_pf->charge() == 0 and abs(c_pf->pdgId()) == 2) {
      pdgid_ = 6.0;
    } else {
      pdgid_ = 7.0;
    }
    c_pf_features.pdgID = pdgid_;
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
                               ChargedCandidateFeatures& c_pf_features,
                               const bool flip,
                               const float distminpfcandsv) {
    commonCandidateToFeatures(
        c_pf, jet, track_info, isWeightedJet, drminpfcandsv, jetR, puppiw, c_pf_features, flip, distminpfcandsv);

    c_pf_features.vtx_ass = vtx_ass_from_pfcand(*c_pf, pv_ass_quality, pv);
    c_pf_features.puppiw = puppiw;

    const auto& pseudo_track = (c_pf->bestTrack()) ? *c_pf->bestTrack() : reco::Track();
    c_pf_features.chi2 = catch_infs_and_bound(std::floor(pseudo_track.normalizedChi2()), 300, -1, 300);
    c_pf_features.quality = quality_from_pfcand(*c_pf);

    // To be implemented if FatJet tag becomes RECO compatible
    // const auto *trk =
    // float dz =
    // float dxy =

    // c_pf_features.dxy =
    // c_pf_features.dz =
    // c_pf_features.dxysig =
    // c_pf_features.dzsig =
  }

}  // namespace btagbtvdeep
