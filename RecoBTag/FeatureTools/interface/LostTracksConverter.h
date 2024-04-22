#ifndef RecoBTag_FeatureTools_LostTracksConverter_h
#define RecoBTag_FeatureTools_LostTracksConverter_h

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "DataFormats/BTauReco/interface/LostTracksFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace btagbtvdeep {

  template <typename CandidateType>
  void commonCandidateToFeatures(const CandidateType* c_pf,
                                 const reco::Jet& jet,
                                 const TrackInfoBuilder& track_info,
                                 const bool& isWeightedJet,
                                 const float& drminpfcandsv,
                                 const float& jetR,
                                 const float& puppiw,
                                 LostTracksFeatures& lt_features,
                                 const bool flip = false,
                                 const float& distminpfcandsv = 0) {
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

    lt_features.btagPf_trackEtaRel = catch_infs_and_bound(track_info.getTrackEtaRel(), 0, -5, 15);
    lt_features.btagPf_trackPtRel = catch_infs_and_bound(track_info.getTrackPtRel(), 0, -1, 4);
    lt_features.btagPf_trackPPar = catch_infs_and_bound(track_info.getTrackPPar(), 0, -1e5, 1e5);
    lt_features.btagPf_trackDeltaR = catch_infs_and_bound(track_info.getTrackDeltaR(), 0, -5, 5);
    lt_features.btagPf_trackPParRatio = catch_infs_and_bound(track_info.getTrackPParRatio(), 0, -10, 100);
    lt_features.btagPf_trackSip3dVal = catch_infs_and_bound(trackSip3dVal, 0, -1, 1e5);
    lt_features.btagPf_trackSip3dSig = catch_infs_and_bound(trackSip3dSig, 0, -1, 4e4);
    lt_features.btagPf_trackSip2dVal = catch_infs_and_bound(trackSip2dVal, 0, -1, 70);
    lt_features.btagPf_trackSip2dSig = catch_infs_and_bound(trackSip2dSig, 0, -1, 4e4);
    lt_features.btagPf_trackJetDistVal = catch_infs_and_bound(track_info.getTrackJetDistVal(), 0, -20, 1);

    lt_features.drminsv = catch_infs_and_bound(drminpfcandsv, 0, -1. * jetR, 0, -1. * jetR);

    lt_features.pt = c_pf->pt();
    lt_features.eta = c_pf->eta();
    lt_features.phi = c_pf->phi();
    lt_features.e = c_pf->energy();
  }

  void packedCandidateToFeatures(const pat::PackedCandidate* c_pf,
                                 const pat::Jet& jet,
                                 const TrackInfoBuilder& track_info,
                                 const bool isWeightedJet,
                                 const float drminpfcandsv,
                                 const float jetR,
                                 const float puppiw,
                                 LostTracksFeatures& lt_features,
                                 const bool flip = false,
                                 const float distminpfcandsv = 0);

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
                               const bool flip = false,
                               const float distminpfcandsv = 0);

}  // namespace btagbtvdeep

#endif  //RecoBTag_FeatureTools_LostTracksConverter_h
