#ifndef DataFormats_BTauReco_DeepDoubleXFeatures_h
#define DataFormats_BTauReco_DeepDoubleXFeatures_h

#include <vector>

#include "DataFormats/BTauReco/interface/JetFeatures.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"
#include "DataFormats/BTauReco/interface/ChargedCandidateFeatures.h"

namespace btagbtvdeep {

class DeepDoubleXFeatures {

  public:

    bool empty() const {
      return is_empty_;
    }  
  
    void filled(){
      is_empty_ = false;
    } 

    JetFeatures jet_features;
    BoostedDoubleSVTagInfoFeatures tag_info_features;

    std::vector<SecondaryVertexFeatures> sv_features;

    std::vector<ChargedCandidateFeatures> c_pf_features;

    std::size_t npv; // used by deep flavour     

  private:
    bool is_empty_ = true;

};    

}  

#endif //DataFormats_BTauReco_DeepDoubleXFeatures_h
