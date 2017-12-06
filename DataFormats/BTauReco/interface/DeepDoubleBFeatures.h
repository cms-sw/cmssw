#ifndef DataFormats_BTauReco_DeepDoubleBFeatures_h
#define DataFormats_BTauReco_DeepDoubleBFeatures_h

#include <vector>

#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

namespace btagbtvdeep {

class DeepDoubleBFeatures {

  public:
    BoostedDoubleSVTagInfoFeatures tag_info_features;

    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<ChargedCandidateFeatures> c_pf_features;
    
};    


}  

#endif //DataFormats_BTauReco_DeepDoubleBFeatures_h
