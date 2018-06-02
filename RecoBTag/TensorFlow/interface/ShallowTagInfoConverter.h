#ifndef RecoBTag_TensorFlow_ShallowTagInfoConverter_h
#define RecoBTag_TensorFlow_ShallowTagInfoConverter_h

#include "RecoBTag/TensorFlow/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfoFeatures.h"

#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace btagbtvdeep {
  
  void bTagToFeatures(const reco::TaggingVariableList & tag_info_vars,
		      ShallowTagInfoFeatures & tag_info_features);
  
}

#endif //RecoBTag_TensorFlow_ShallowTagInfoConverter_h


