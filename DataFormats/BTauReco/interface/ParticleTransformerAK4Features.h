#ifndef DataFormats_BTauReco_ParticleTransformerAK4Features_h
#define DataFormats_BTauReco_ParticleTransformerAK4Features_h

//#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

#include <vector>

namespace btagbtvdeep {

  class ParticleTransformerAK4Features {
  public:
    //JetFeatures jet_features;

    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<NeutralCandidateFeatures> n_pf_features;
    std::vector<ChargedCandidateFeatures> c_pf_features;

    //std::size_t npv;  // used by deep flavour
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_ParticleTransformerAK4Features_h