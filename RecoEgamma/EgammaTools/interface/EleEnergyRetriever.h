#ifndef RecoEgamma_EgammaTools_EleEnergyRetriever_h
#define RecoEgamma_EgammaTools_EleEnergyRetriever_h

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "FWCore/Utilities/interface/Exception.h"

class EleEnergyRetriever {
public:
  enum class EnergyType{EcalTrk,Ecal,SuperCluster,SuperClusterRaw};
  
  EleEnergyRetriever(const std::string& typeStr):
    type_(convertFromStr(typeStr)){}
       
   static EnergyType convertFromStr(const std::string& typeStr){
     if(typeStr=="EcalTrk") return EnergyType::EcalTrk;
     else if(typeStr=="Ecal") return EnergyType::Ecal;
     else if(typeStr=="SuperCluster") return EnergyType::SuperCluster;
     else if(typeStr=="SuperClusterRaw") return EnergyType::SuperClusterRaw;
     else {
       throw cms::Exception("ConfigError") <<" type \""<<typeStr<<"\" not recognised, must be of type EcalTrk,Ecal,SuperCluster,SuperClusterRaw";
     }
   }

  float operator()(const reco::GsfElectron& ele)const{
    switch(type_){
    case EnergyType::EcalTrk: return ele.energy();
    case EnergyType::Ecal: return ele.ecalEnergy();
    case EnergyType::SuperCluster: return ele.superCluster()->energy();
    case EnergyType::SuperClusterRaw: return ele.superCluster()->rawEnergy();
    }
    return 0.;
  }

private:
  EnergyType type_;

};

#endif
