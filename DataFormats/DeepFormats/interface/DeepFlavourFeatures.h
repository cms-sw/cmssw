#ifndef DataFormats_DeepFormats_DeepFlavourFeatures_h
#define DataFormats_DeepFormats_DeepFlavourFeatures_h

#include <vector>

#include "DataFormats/DeepFormats/interface/JetFeatures.h"
#include "DataFormats/DeepFormats/interface/SecondaryVertexFeatures.h"

namespace deep {

class DeepFlavourFeatures {

  public:
    JetFeatures jet_features;
    
    std::vector<SecondaryVertexFeatures> secondary_vertex_features;
    
};    


}  

#endif //DataFormats_DeepFormats_DeepFlavourFeatures_h
