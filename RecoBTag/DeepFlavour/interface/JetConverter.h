#ifndef RecoBTag_DeepFlavour_JetConverter_h
#define RecoBTag_DeepFlavour_JetConverter_h

namespace btagbtvdeep {

  class JetConverter {
    public:

      template <typename JetType, typename JetFeaturesType>
      static void JetToFeatures( const JetType & jet, JetFeaturesType & jet_features) {
        jet_features.pt = jet.pt(); // uncorrected was used on DeepNtuples
        jet_features.eta = jet.eta();
        jet_features.phi = jet.phi();
        jet_features.corr_pt = jet.pt();
        jet_features.mass = jet.mass();
        jet_features.energy = jet.energy();
      }
  }; 

}

#endif //RecoBTag_DeepFlavour_JetConverter_h

