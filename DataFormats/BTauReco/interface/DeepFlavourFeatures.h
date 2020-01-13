#ifndef DataFormats_BTauReco_DeepFlavourFeatures_h
#define DataFormats_BTauReco_DeepFlavourFeatures_h

#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/NeutralCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"
#include "DataFormats/BTauReco/interface/SeedingTrackFeatures.h"

#include <vector>

namespace btagbtvdeep {

  class DeepFlavourFeatures {
  public:
    JetFeatures jet_features;
    ShallowTagInfoFeatures tag_info_features;

    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<NeutralCandidateFeatures> n_pf_features;
    std::vector<ChargedCandidateFeatures> c_pf_features;

    std::vector<SeedingTrackFeatures> seed_features;

    std::size_t npv;  // used by deep flavour
  };

}  // namespace btagbtvdeep

#endif  //DataFormats_BTauReco_DeepFlavourFeatures_h
