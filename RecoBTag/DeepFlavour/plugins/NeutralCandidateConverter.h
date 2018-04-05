#ifndef RecoSV_DeepFlavour_NeutralCandidateConverter_h
#define RecoSV_DeepFlavour_NeutralCandidateConverter_h

#include "deep_helpers.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"

namespace btagbtvdeep {

  class NeutralCandidateConverter {
    public:

      template <typename CandidateType>
      static void CommonCandidateToFeatures(const CandidateType * n_pf,
                                            const reco::Jet & jet,
                                            const float & drminpfcandsv,
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

      static void PackedCandidateToFeatures(const pat::PackedCandidate * n_pf,
                                            const pat::Jet & jet,
                                            const float drminpfcandsv,
                                            NeutralCandidateFeatures & n_pf_features) {

        CommonCandidateToFeatures(n_pf, jet, drminpfcandsv, n_pf_features);

        n_pf_features.hadFrac = n_pf->hcalFraction();
        n_pf_features.puppiw = n_pf->puppiWeight();
    
      } 
    
      static void RecoCandidateToFeatures(const reco::PFCandidate * n_pf,
                                          const reco::Jet & jet,
                                          const float drminpfcandsv, const float puppiw,
                                          NeutralCandidateFeatures & n_pf_features) {

        CommonCandidateToFeatures(n_pf, jet, drminpfcandsv, n_pf_features);
        n_pf_features.puppiw = puppiw;

        // need to get a value map and more stuff to do properly
        // otherwise will be different than for PackedCandidates
        // https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/PatAlgos/python/slimming/packedPFCandidates_cfi.py
        if(abs(n_pf->pdgId()) == 1 || abs(n_pf->pdgId()) == 130) {
          n_pf_features.hadFrac = n_pf->hcalEnergy()/(n_pf->ecalEnergy()+n_pf->hcalEnergy());
        } else {
          n_pf_features.hadFrac = 0;
        }
    
      } 

  };

}

#endif //RecoSV_DeepFlavour_NeutralCandidateConverter_h
