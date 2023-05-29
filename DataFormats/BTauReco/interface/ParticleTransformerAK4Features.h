#ifndef DataFormats_BTauReco_ParticleTransformerAK4Features_h
#define DataFormats_BTauReco_ParticleTransformerAK4Features_h

#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

#include <vector>

namespace btagbtvdeep {

  class ParticleTransformerAK4Features {
  public:
    bool is_filled = true;
    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<NeutralCandidateFeatures> n_pf_features;
    std::vector<ChargedCandidateFeatures> c_pf_features;
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_ParticleTransformerAK4Features_h
