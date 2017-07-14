#ifndef RecoBTag_DeepFlavour_jet_features_converter_h
#define RecoBTag_DeepFlavour_jet_features_converter_h

namespace deep {

  template <typename JetType, typename JetFeaturesType>
  void jet_features_converter( const JetType & jet, JetFeaturesType & jet_features) {

    jet_features.pt = jet.correctedJet("Uncorrected").pt();
    jet_features.eta = jet.eta();
    jet_features.phi = jet.phi();
    jet_features.corr_pt = jet.pt();
    jet_features.mass = jet.mass();
    jet_features.energy = jet.energy();


    //https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVRun2016
    try{
      float NHF  = jet.neutralHadronEnergyFraction();
      float NEMF = jet.neutralEmEnergyFraction();
      float CHF  = jet.chargedHadronEnergyFraction();
      float CEMF = jet.chargedEmEnergyFraction();
      float NumConst = jet.chargedMultiplicity()+jet.neutralMultiplicity();
      float NumNeutralParticles =jet.neutralMultiplicity();
      float CHM      = jet.chargedMultiplicity();

      jet_features.looseId = ((NHF<0.99 && NEMF<0.99 && NumConst>1) &&
                             ((abs(jet_features.eta)<=2.4 && CHF>0 && CHM>0 && CEMF<0.99) ||
                             abs(jet_features.eta)>2.4) && abs(jet_features.eta)<=2.7) ||
                             (NHF<0.98 && NEMF>0.01 && NumNeutralParticles>2 &&
                             abs(jet_features.eta)>2.7 && abs(jet_features.eta)<=3.0 ) ||
                             (NEMF<0.90 && NumNeutralParticles>10 && abs(jet_features.eta)>3.0 );
    }catch(const cms::Exception &e){
      jet_features.looseId = 1;
    }

  } 
 


}

#endif

