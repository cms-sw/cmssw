#ifndef RecoBTag_TensorFlow_BoostedDoubleSVTagInfoConverter_h
#define RecoBTag_TensorFlow_BoostedDoubleSVTagInfoConverter_h

#include "RecoBTag/TensorFlow/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"

#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace btagbtvdeep {
  
  void doubleBTagToFeatures(const reco::TaggingVariableList & tag_info_vars,
			    BoostedDoubleSVTagInfoFeatures & tag_info_features) ;
  
}

#endif //RecoBTag_TensorFlow_BoostedDoubleSVTagInfoConverter_h


