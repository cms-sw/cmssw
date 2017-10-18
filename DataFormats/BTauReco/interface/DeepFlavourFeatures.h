#ifndef DataFormats_DeepFormats_DeepFlavourFeatures_h
#define DataFormats_DeepFormats_DeepFlavourFeatures_h

#include <vector>

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/DeepFormats/interface/JetFeatures.h"
#include "DataFormats/DeepFormats/interface/SecondaryVertexFeatures.h"
#include "DataFormats/DeepFormats/interface/ShallowTagInfoFeatures.h"
#include "DataFormats/DeepFormats/interface/NeutralCandidateFeatures.h"
#include "DataFormats/DeepFormats/interface/ChargedCandidateFeatures.h"

namespace btagbtvdeep {

class DeepFlavourFeatures {

  public:
    JetFeatures jet_features;
    ShallowTagInfoFeatures tag_info_features;
    
    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<NeutralCandidateFeatures> n_pf_features;
    std::vector<ChargedCandidateFeatures> c_pf_features;
    
    std::size_t npv; // used by deep flavour

};    


}  

#endif //DataFormats_DeepFormats_DeepFlavourFeatures_h
