#ifndef DataFormats_BTauReco_DeepDoubleCvLFeatures_h
#define DataFormats_BTauReco_DeepDoubleCvLFeatures_h

#include <vector>

#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

namespace btagbtvdeep {

class DeepDoubleCvLFeatures {

  public:
    JetFeatures jet_features;
    BoostedDoubleSVTagInfoFeatures tag_info_features;

    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<ChargedCandidateFeatures> c_pf_features;

    std::size_t npv; // used by deep flavour     
};    


}  

#endif //DataFormats_BTauReco_DeepDoubleCvLFeatures_h
