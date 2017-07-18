#ifndef DataFormats_DeepFormats_DeepFlavourFeatures_h
#define DataFormats_DeepFormats_DeepFlavourFeatures_h

#include <vector>

#include "DataFormats/DeepFormats/interface/JetFeatures.h"
#include "DataFormats/DeepFormats/interface/SecondaryVertexFeatures.h"
#include "DataFormats/DeepFormats/interface/ShallowTagInfoFeatures.h"

namespace deep {

class DeepFlavourFeatures {

  public:
    JetFeatures jet_features;
    ShallowTagInfoFeatures tag_info_features;
    
    std::vector<SecondaryVertexFeatures> sv_features;
    
};    


}  

#endif //DataFormats_DeepFormats_DeepFlavourFeatures_h
