#ifndef RecoBTag_FeatureTools_NeutralCandidateConverter_h
#define RecoBTag_FeatureTools_NeutralCandidateConverter_h

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

namespace btagbtvdeep {

  void packedCandidateToFeatures(const pat::PackedCandidate* n_pf,
                                 const pat::Jet& jet,
                                 const float drminpfcandsv,
                                 const float jetR,
                                 const float puppiw,
                                 NeutralCandidateFeatures& n_pf_features);

  void recoCandidateToFeatures(const reco::PFCandidate* n_pf,
                               const reco::Jet& jet,
                               const float drminpfcandsv,
                               const float jetR,
                               const float puppiw,
                               NeutralCandidateFeatures& n_pf_features);

  template <typename CandidateType>
  static void commonCandidateToFeatures(const CandidateType* n_pf,
                                        const reco::Jet& jet,
                                        const float& drminpfcandsv,
                                        const float& jetR,
                                        NeutralCandidateFeatures& n_pf_features) {
    std::pair<float, float> drSubjetFeatures = getDRSubjetFeatures(jet, n_pf);
    n_pf_features.drsubjet1 = drSubjetFeatures.first;
    n_pf_features.drsubjet2 = drSubjetFeatures.second;

    // Jet relative vars
    n_pf_features.ptrel = catch_infs_and_bound(n_pf->pt() / jet.pt(), 0, -1, 0, -1);
    n_pf_features.ptrel_noclip = n_pf->pt() / jet.pt();
    n_pf_features.deltaR = catch_infs_and_bound(reco::deltaR(*n_pf, jet), 0, -0.6, 0, -0.6);
    n_pf_features.deltaR_noclip = reco::deltaR(*n_pf, jet);
    n_pf_features.erel = n_pf->energy() / jet.energy();
    n_pf_features.isGamma = 0;
    if (std::abs(n_pf->pdgId()) == 22)
      n_pf_features.isGamma = 1;

    n_pf_features.drminsv = catch_infs_and_bound(drminpfcandsv, 0, -1. * jetR, 0, -1. * jetR);
  }

}  // namespace btagbtvdeep

#endif  //RecoBTag_FeatureTools_NeutralCandidateConverter_h
