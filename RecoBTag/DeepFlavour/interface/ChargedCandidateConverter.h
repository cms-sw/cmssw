#ifndef RecoSV_DeepFlavour_ChargedCandidateConverter_h
#define RecoSV_DeepFlavour_ChargedCandidateConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "RecoBTag/DeepFlavour/interface/TrackInfoBuilder.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace btagbtvdeep {

      // conversion map from quality flags used in PV association and miniAOD one
      //constexpr static int qualityMap[8]  = {1,0,1,1,4,4,5,6};

      /*enum qualityFlagsShiftsAndMasks {
            assignmentQualityMask = 0x7, 
            assignmentQualityShift = 0,
            trackHighPurityMask  = 0x8, 
            trackHighPurityShift=3,
            lostInnerHitsMask = 0x30, 
            lostInnerHitsShift=4,
            muonFlagsMask = 0x0600, 
            muonFlagsShift=9
      };*/

      template <typename CandidateType>
      void CommonCandidateToFeatures(const CandidateType * c_pf,
                                            const reco::Jet & jet,
                                            const TrackInfoBuilder & track_info,
                                            const float & drminpfcandsv,
                                            ChargedCandidateFeatures & c_pf_features) {

        c_pf_features.ptrel = catch_infs_and_bound(c_pf->pt()/jet.pt(),
                                                          0,-1,0,-1);

        c_pf_features.btagPf_trackEtaRel     =catch_infs_and_bound(track_info.getTrackEtaRel(),  0,-5,15);
        c_pf_features.btagPf_trackPtRel      =catch_infs_and_bound(track_info.getTrackPtRel(),   0,-1,4);
        c_pf_features.btagPf_trackPPar       =catch_infs_and_bound(track_info.getTrackPPar(),    0,-1e5,1e5 );
        c_pf_features.btagPf_trackDeltaR     =catch_infs_and_bound(track_info.getTrackDeltaR(),  0,-5,5 );
        c_pf_features.btagPf_trackPtRatio    =catch_infs_and_bound(track_info.getTrackPtRatio(), 0,-1,10);
        c_pf_features.btagPf_trackPParRatio  =catch_infs_and_bound(track_info.getTrackPParRatio(),0,-10,100);
        c_pf_features.btagPf_trackSip3dVal   =catch_infs_and_bound(track_info.getTrackSip3dVal(), 0, -1,1e5 );
        c_pf_features.btagPf_trackSip3dSig   =catch_infs_and_bound(track_info.getTrackSip3dSig(), 0, -1,4e4 );
        c_pf_features.btagPf_trackSip2dVal   =catch_infs_and_bound(track_info.getTrackSip2dVal(), 0, -1,70 );
        c_pf_features.btagPf_trackSip2dSig   =catch_infs_and_bound(track_info.getTrackSip2dSig(), 0, -1,4e4 );
        c_pf_features.btagPf_trackJetDistVal =catch_infs_and_bound(track_info.getTrackJetDistVal(),0,-20,1 );

        c_pf_features.drminsv = catch_infs_and_bound(drminpfcandsv,0,-0.8,0,-0.8);

      }
    
      void PackedCandidateToFeatures(const pat::PackedCandidate * c_pf,
                                            const pat::Jet & jet,
                                            const TrackInfoBuilder & track_info,
                                            const float drminpfcandsv,
                                            ChargedCandidateFeatures & c_pf_features) ;

    
      void RecoCandidateToFeatures(const reco::PFCandidate * c_pf,
                                          const reco::Jet & jet,
                                          const TrackInfoBuilder & track_info,
                                          const float drminpfcandsv, const float puppiw,
                                          const int pv_ass_quality,
                                          const reco::VertexRef & pv, 
                                          ChargedCandidateFeatures & c_pf_features) ;


  // static data member (avoid undefined ref)
//  constexpr int ChargedCandidateConverter::qualityMap[8]; 


}

#endif //RecoSV_DeepFlavour_ChargedCandidateConverter_h
