#include "RecoBTag/TensorFlow/interface/ChargedCandidateConverter.h"


namespace btagbtvdeep {
  
  constexpr static int qualityMap[8]  = {1,0,1,1,4,4,5,6};      
  
  
  enum qualityFlagsShiftsAndMasks {
    assignmentQualityMask = 0x7, 
    assignmentQualityShift = 0,
    trackHighPurityMask  = 0x8, 
    trackHighPurityShift=3,
    lostInnerHitsMask = 0x30, 
    lostInnerHitsShift=4,
    muonFlagsMask = 0x0600, 
    muonFlagsShift=9
  };
  
  void packedCandidateToFeatures(const pat::PackedCandidate * c_pf,
				 const pat::Jet & jet,
				 const TrackInfoBuilder & track_info,
				 const float drminpfcandsv, const float jetR,
				 ChargedCandidateFeatures & c_pf_features,
				 const bool flip) {
    
    commonCandidateToFeatures(c_pf, jet, track_info, drminpfcandsv, jetR, c_pf_features, flip);
    
    c_pf_features.vtx_ass = c_pf->pvAssociationQuality();
    
    c_pf_features.puppiw = c_pf->puppiWeight();
    
    // if PackedCandidate does not have TrackDetails this gives an Exception
    // because unpackCovariance might be called for pseudoTrack/bestTrack
    if (c_pf->hasTrackDetails()) {
      const auto & pseudo_track =  c_pf->pseudoTrack();
      c_pf_features.chi2 = catch_infs_and_bound(pseudo_track.normalizedChi2(),300,-1,300);
      // this returns the quality enum not a mask.
      c_pf_features.quality = pseudo_track.qualityMask();
    } else {
      // default negative chi2 and loose track if notTrackDetails
      c_pf_features.chi2 = catch_infs_and_bound(-1,300,-1,300);
      c_pf_features.quality =(1 << reco::TrackBase::loose);
    }
    
  } 
    
  void recoCandidateToFeatures(const reco::PFCandidate * c_pf,
			       const reco::Jet & jet,
			       const TrackInfoBuilder & track_info,
			       const float drminpfcandsv, const float jetR, const float puppiw,
			       const int pv_ass_quality,
			       const reco::VertexRef & pv, 
			       ChargedCandidateFeatures & c_pf_features,
			       const bool flip) {
    
    commonCandidateToFeatures(c_pf, jet, track_info, drminpfcandsv, jetR, c_pf_features, flip);
    
    c_pf_features.vtx_ass = (float) pat::PackedCandidate::PVAssociationQuality(qualityMap[pv_ass_quality]);
    if (c_pf->trackRef().isNonnull() && 
	pv->trackWeight(c_pf->trackRef()) > 0.5 &&
	pv_ass_quality == 7) {
      c_pf_features.vtx_ass = (float) pat::PackedCandidate::UsedInFitTight;
    }
    
    c_pf_features.puppiw = puppiw;
    
    const auto & pseudo_track =  (c_pf->bestTrack()) ? *c_pf->bestTrack() : reco::Track();
    c_pf_features.chi2 = catch_infs_and_bound(std::floor(pseudo_track.normalizedChi2()),300,-1,300);
    // conditions from PackedCandidate producer
    bool highPurity = c_pf->trackRef().isNonnull() && pseudo_track.quality(reco::Track::highPurity);
    // do same bit operations than in PackedCandidate
    uint16_t qualityFlags = 0;
    qualityFlags = (qualityFlags & ~trackHighPurityMask) | ((highPurity << trackHighPurityShift) & trackHighPurityMask);
    bool isHighPurity = (qualityFlags & trackHighPurityMask)>>trackHighPurityShift;
    // to do as in TrackBase
    uint8_t quality = (1 << reco::TrackBase::loose);
    if (isHighPurity) {
      quality |= (1 << reco::TrackBase::highPurity);
    } 
    c_pf_features.quality = quality; 
    
  } 
  
}

