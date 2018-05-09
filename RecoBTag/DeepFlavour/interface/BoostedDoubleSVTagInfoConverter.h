#ifndef RecoBTag_DeepFlavour_BoostedDoubleSVTagInfoConverter_h
#define RecoBTag_DeepFlavour_BoostedDoubleSVTagInfoConverter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"

#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace btagbtvdeep {
  
  void doubleBTagToFeatures(const reco::TaggingVariableList & tag_info_vars,
			    BoostedDoubleSVTagInfoFeatures & tag_info_features) ;
  
}

#endif //RecoBTag_DeepFlavour_BoostedDoubleSVTagInfoConverter_h


