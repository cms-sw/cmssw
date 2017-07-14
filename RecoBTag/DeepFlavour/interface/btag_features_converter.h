#ifndef RecoBTag_DeepFlavour_btag_features_converter_h
#define RecoBTag_DeepFlavour_btag_features_converter_h

namespace deep {

  template <typename TagInfoType, typename BTagFeaturesType>
  void btag_features_converter( const TagInfoType & tag_info, BTagFeaturesType & btag_features) {

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
