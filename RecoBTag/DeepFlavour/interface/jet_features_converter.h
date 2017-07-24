#ifndef RecoBTag_DeepFlavour_jet_features_converter_h
#define RecoBTag_DeepFlavour_jet_features_converter_h

namespace deep {

  template <typename JetType, typename JetFeaturesType>
  void jet_features_converter( const JetType & jet, JetFeaturesType & jet_features) {

    jet_features.pt = jet.pt(); // uncorrected was used on DeepNtuples
    jet_features.eta = jet.eta();
    jet_features.phi = jet.phi();
    jet_features.corr_pt = jet.pt();
    jet_features.mass = jet.mass();
    jet_features.energy = jet.energy();

  } 

}

#endif

