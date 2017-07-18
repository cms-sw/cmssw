#ifndef RecoBTag_DeepFlavour_btag_features_converter_h
#define RecoBTag_DeepFlavour_btag_features_converter_h

namespace deep {

  static constexpr std::size_t max_jetNSelectedTracks =100;

  template <typename TagInfoVarsType, typename TagInfoFeaturesType>
  void btag_features_converter( const TagInfoVarsType & tag_info_vars, TagInfoFeaturesType & tag_info_features) {

    tag_info_features.trackJetPt                 = tag_info_vars.get(reco::btau::trackJetPt, -999);
    tag_info_features.jetNSecondaryVertices      = tag_info_vars.get(reco::btau::jetNSecondaryVertices, -1);
    tag_info_features.trackSumJetEtRatio         = tag_info_vars.get(reco::btau::trackSumJetEtRatio, -999);
    tag_info_features.trackSumJetDeltaR          = tag_info_vars.get(reco::btau::trackSumJetDeltaR, -999);
    tag_info_features.vertexCategory             = tag_info_vars.get(reco::btau::vertexCategory, -999);
    tag_info_features.trackSip2dValAboveCharm    = tag_info_vars.get(reco::btau::trackSip2dValAboveCharm, -999);
    tag_info_features.trackSip2dSigAboveCharm    = tag_info_vars.get(reco::btau::trackSip2dSigAboveCharm, -999);
    tag_info_features.trackSip3dValAboveCharm    = tag_info_vars.get(reco::btau::trackSip3dValAboveCharm, -999);
    tag_info_features.trackSip3dSigAboveCharm    = tag_info_vars.get(reco::btau::trackSip3dSigAboveCharm, -999);
    tag_info_features.jetNTracks = tag_info_vars.get(reco::btau::jetNTracks, -1);
    tag_info_features.jetNTracksEtaRel = tag_info_vars.get(reco::btau::jetNTracksEtaRel, -1);
    tag_info_features.jetNSelectedTracks = std::min(tag_info_vars.getList(reco::btau::trackMomentum, false).size(),
                                                    max_jetNSelectedTracks);

  } 

  template <typename TagInfoType, typename JetType>
  const TagInfoType * get_tag_info_from_jet( const JetType & jet, const std::string & tag_info_name) {

    if(!jet.hasTagInfo(tag_info_name)) {
        std::stringstream name_stream;
        for(auto &lab : jet.tagInfoLabels())
            name_stream << lab << ", ";
        throw cms::Exception("ValueError") << "There is no tagInfo embedded in the jet named: " <<
           tag_info_name << ". The available ones are: " << name_stream.str() << std::endl;
    }
    return dynamic_cast<const TagInfoType*>(jet.tagInfo(tag_info_name));
  } 
 


}

#endif //RecoBTag_DeepFlavour_btag_features_converter_h
