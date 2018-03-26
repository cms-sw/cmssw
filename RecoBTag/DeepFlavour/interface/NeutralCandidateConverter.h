#ifndef RecoSV_DeepFlavour_NeutralCandidateConverter_h
#define RecoSV_DeepFlavour_NeutralCandidateConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


namespace btagbtvdeep {



      void PackedCandidateToFeatures(const pat::PackedCandidate * n_pf,
				     const pat::Jet & jet,
				     const float drminpfcandsv, const double jetR,
				     NeutralCandidateFeatures & n_pf_features) ;

    
      void RecoCandidateToFeatures(const reco::PFCandidate * n_pf,
				   const reco::Jet & jet,
				   const float drminpfcandsv, const double jetR, const float puppiw,
				   NeutralCandidateFeatures & n_pf_features) ;


      template <typename CandidateType>
      static void CommonCandidateToFeatures(const CandidateType * n_pf,
                                            const reco::Jet & jet,
                                            const float & drminpfcandsv, const double & jetR,
                                            NeutralCandidateFeatures & n_pf_features) {

        n_pf_features.ptrel = catch_infs_and_bound(n_pf->pt()/jet.pt(),
                                                   0,-1,0,-1);
        n_pf_features.deltaR = catch_infs_and_bound(reco::deltaR(*n_pf,jet),
                                                    0,-0.6,0,-0.6);
        n_pf_features.isGamma = 0;
        if(std::abs(n_pf->pdgId())==22)  n_pf_features.isGamma = 1;


        n_pf_features.drminsv = catch_infs_and_bound(drminpfcandsv,
                                                     0,-0.4,0,-0.4);

      }



}

#endif //RecoSV_DeepFlavour_NeutralCandidateConverter_h
