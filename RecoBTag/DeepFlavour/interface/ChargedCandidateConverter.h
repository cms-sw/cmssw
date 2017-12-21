#ifndef RecoSV_DeepFlavour_ChargedCandidateConverter_h
#define RecoSV_DeepFlavour_ChargedCandidateConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "RecoBTag/DeepFlavour/interface/TrackInfoBuilder.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace btagbtvdeep {

  
  class ChargedCandidateConverter {

    public:

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
      static void CommonCandidateToFeatures(const CandidateType * c_pf,
                                            const reco::Jet & jet,
                                            const TrackInfoBuilder & track_info,
                                            const float & drminpfcandsv,
                                            ChargedCandidateFeatures & c_pf_features) ;

    
      static void PackedCandidateToFeatures(const pat::PackedCandidate * c_pf,
                                            const pat::Jet & jet,
                                            const TrackInfoBuilder & track_info,
                                            const float drminpfcandsv,
                                            ChargedCandidateFeatures & c_pf_features) ;

    
      static void RecoCandidateToFeatures(const reco::PFCandidate * c_pf,
                                          const reco::Jet & jet,
                                          const TrackInfoBuilder & track_info,
                                          const float drminpfcandsv, const float puppiw,
                                          const int pv_ass_quality,
                                          const reco::VertexRef & pv, 
                                          ChargedCandidateFeatures & c_pf_features) ;

  };

  // static data member (avoid undefined ref)
//  constexpr int ChargedCandidateConverter::qualityMap[8]; 


}

#endif //RecoSV_DeepFlavour_ChargedCandidateConverter_h
