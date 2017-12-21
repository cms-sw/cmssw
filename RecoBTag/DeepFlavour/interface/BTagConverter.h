#ifndef RecoBTag_DeepFlavour_BTagConverter_h
#define RecoBTag_DeepFlavour_BTagConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfoFeatures.h"

#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace btagbtvdeep {

  class BTagConverter {
    public:

      static void BTagToFeatures(const reco::TaggingVariableList & tag_info_vars,
                                 ShallowTagInfoFeatures & tag_info_features) ;
    
  };

}

#endif //RecoBTag_DeepFlavour_BTagConverter_h


