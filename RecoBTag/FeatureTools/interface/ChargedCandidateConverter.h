#ifndef RecoBTag_FeatureTools_ChargedCandidateConverter_h
#define RecoBTag_FeatureTools_ChargedCandidateConverter_h

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace btagbtvdeep {

  template <typename CandidateType>
  void commonCandidateToFeatures(const CandidateType* c_pf,
                                 const reco::Jet& jet,
                                 const TrackInfoBuilder& track_info,
                                 const float& drminpfcandsv,
                                 const float& jetR,
                                 ChargedCandidateFeatures& c_pf_features,
                                 const bool flip = false) {
    float trackSip2dVal = track_info.getTrackSip2dVal();
    float trackSip2dSig = track_info.getTrackSip2dSig();
    float trackSip3dVal = track_info.getTrackSip3dVal();
    float trackSip3dSig = track_info.getTrackSip3dSig();
    if (flip == true) {
      trackSip2dVal = -trackSip2dVal;
      trackSip2dSig = -trackSip2dSig;
      trackSip3dSig = -trackSip3dSig;
      trackSip3dVal = -trackSip3dVal;
    }

    c_pf_features.deltaR = reco::deltaR(*c_pf, jet);
    c_pf_features.ptrel = catch_infs_and_bound(c_pf->pt() / jet.pt(), 0, -1, 0, -1);
    c_pf_features.ptrel_noclip = c_pf->pt() / jet.pt();
    c_pf_features.erel = c_pf->energy() / jet.energy();
    const float etasign = jet.eta() > 0 ? 1 : -1;
    c_pf_features.etarel = etasign * (c_pf->eta() - jet.eta());

    c_pf_features.btagPf_trackEtaRel = catch_infs_and_bound(track_info.getTrackEtaRel(), 0, -5, 15);
    c_pf_features.btagPf_trackPtRel = catch_infs_and_bound(track_info.getTrackPtRel(), 0, -1, 4);
    c_pf_features.btagPf_trackPPar = catch_infs_and_bound(track_info.getTrackPPar(), 0, -1e5, 1e5);
    c_pf_features.btagPf_trackDeltaR = catch_infs_and_bound(track_info.getTrackDeltaR(), 0, -5, 5);
    c_pf_features.btagPf_trackPtRatio = catch_infs_and_bound(track_info.getTrackPtRatio(), 0, -1, 10);
    c_pf_features.btagPf_trackPParRatio = catch_infs_and_bound(track_info.getTrackPParRatio(), 0, -10, 100);
    c_pf_features.btagPf_trackSip3dVal = catch_infs_and_bound(trackSip3dVal, 0, -1, 1e5);
    c_pf_features.btagPf_trackSip3dSig = catch_infs_and_bound(trackSip3dSig, 0, -1, 4e4);
    c_pf_features.btagPf_trackSip2dVal = catch_infs_and_bound(trackSip2dVal, 0, -1, 70);
    c_pf_features.btagPf_trackSip2dSig = catch_infs_and_bound(trackSip2dSig, 0, -1, 4e4);
    c_pf_features.btagPf_trackJetDistVal = catch_infs_and_bound(track_info.getTrackJetDistVal(), 0, -20, 1);

    c_pf_features.drminsv = catch_infs_and_bound(drminpfcandsv, 0, -1. * jetR, 0, -1. * jetR);

    // subjet related
    const auto* patJet = dynamic_cast<const pat::Jet*>(&jet);
    if (!patJet) {
      throw edm::Exception(edm::errors::InvalidReference) << "Input is not a pat::Jet.";
    }

    if (patJet->nSubjetCollections() > 0) {
      auto subjets = patJet->subjets();
      // sort by pt
      std::sort(subjets.begin(), subjets.end(), [](const edm::Ptr<pat::Jet>& p1, const edm::Ptr<pat::Jet>& p2) {
        return p1->pt() > p2->pt();
      });
      c_pf_features.drsubjet1 = !subjets.empty() ? reco::deltaR(*c_pf, *subjets.at(0)) : -1;
      c_pf_features.drsubjet2 = subjets.size() > 1 ? reco::deltaR(*c_pf, *subjets.at(1)) : -1;
    } else {
      // AK4 jets don't have subjets
      c_pf_features.drsubjet1 = -1;
      c_pf_features.drsubjet2 = -1;
    }
  }

  void packedCandidateToFeatures(const pat::PackedCandidate* c_pf,
                                 const pat::Jet& jet,
                                 const TrackInfoBuilder& track_info,
                                 const float drminpfcandsv,
                                 const float jetR,
                                 ChargedCandidateFeatures& c_pf_features,
                                 const bool flip = false);

  void recoCandidateToFeatures(const reco::PFCandidate* c_pf,
                               const reco::Jet& jet,
                               const TrackInfoBuilder& track_info,
                               const float drminpfcandsv,
                               const float jetR,
                               const float puppiw,
                               const int pv_ass_quality,
                               const reco::VertexRef& pv,
                               ChargedCandidateFeatures& c_pf_features,
                               const bool flip = false);

}  // namespace btagbtvdeep

#endif  //RecoBTag_FeatureTools_ChargedCandidateConverter_h
