#ifndef RecoSV_DeepFlavour_NeutralCandidateConverter_h
#define RecoSV_DeepFlavour_NeutralCandidateConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


namespace btagbtvdeep {

  class NeutralCandidateConverter {
    public:

      template <typename CandidateType>
      static void CommonCandidateToFeatures(const CandidateType * n_pf,
                                            const reco::Jet & jet,
                                            const float & drminpfcandsv,
                                            NeutralCandidateFeatures & n_pf_features) ;


      static void PackedCandidateToFeatures(const pat::PackedCandidate * n_pf,
                                            const pat::Jet & jet,
                                            const float drminpfcandsv,
                                            NeutralCandidateFeatures & n_pf_features) ;

    
      static void RecoCandidateToFeatures(const reco::PFCandidate * n_pf,
                                          const reco::Jet & jet,
                                          const float drminpfcandsv, const float puppiw,
                                          NeutralCandidateFeatures & n_pf_features) ;


  };

}

#endif //RecoSV_DeepFlavour_NeutralCandidateConverter_h
