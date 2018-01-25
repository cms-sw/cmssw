#ifndef RecoBTag_DeepFlavour_BTagConverter_h
#define RecoBTag_DeepFlavour_BTagConverter_h

#include "deep_helpers.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfoFeatures.h"

#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace btagbtvdeep {

  class BTagConverter {
    public:
      static constexpr std::size_t max_jetNSelectedTracks =100;

      static void BTagToFeatures(const reco::TaggingVariableList & tag_info_vars,
                                 ShallowTagInfoFeatures & tag_info_features) {
    
        tag_info_features.trackSumJetEtRatio         = tag_info_vars.get(reco::btau::trackSumJetEtRatio, -999);
        tag_info_features.trackSumJetDeltaR          = tag_info_vars.get(reco::btau::trackSumJetDeltaR, -999);
        tag_info_features.vertexCategory             = tag_info_vars.get(reco::btau::vertexCategory, -999);
        tag_info_features.trackSip2dValAboveCharm    = tag_info_vars.get(reco::btau::trackSip2dValAboveCharm, -999);
        tag_info_features.trackSip2dSigAboveCharm    = tag_info_vars.get(reco::btau::trackSip2dSigAboveCharm, -999);
        tag_info_features.trackSip3dValAboveCharm    = tag_info_vars.get(reco::btau::trackSip3dValAboveCharm, -999);
        tag_info_features.trackSip3dSigAboveCharm    = tag_info_vars.get(reco::btau::trackSip3dSigAboveCharm, -999);
        tag_info_features.jetNTracksEtaRel = tag_info_vars.get(reco::btau::jetNTracksEtaRel, -1);
        tag_info_features.jetNSelectedTracks = std::min(tag_info_vars.getList(reco::btau::trackMomentum, false).size(),
                                                        max_jetNSelectedTracks);
    
      } 
  };

}

#endif //RecoBTag_DeepFlavour_BTagConverter_h


