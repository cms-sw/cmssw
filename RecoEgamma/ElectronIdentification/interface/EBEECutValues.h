#ifndef EgammaAnalysis_ElectronTools_EBEECutValues_h
#define EgammaAnalysis_ElectronTools_EBEECutValues_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace reco {
  typedef edm::Ptr<reco::GsfElectron> GsfElectronPtr;
}

template<typename T>
class EBEECutValuesT {
private:
  T barrel_;
  T endcap_;
  const double barrelCutOff_=1.479; //this is currrently used to identify if object is barrel or endcap but may change
  
public:
  EBEECutValuesT(const edm::ParameterSet& params,const std::string& name):
    barrel_(params.getParameter<T>(name+"EB")),
    endcap_(params.getParameter<T>(name+"EE")){}
  T operator()(const reco::GsfElectronPtr& cand)const{return isBarrel(cand) ? barrel_ : endcap_;}

private:
  const bool isBarrel(const reco::GsfElectronPtr& cand)const{return std::abs(cand->superCluster()->position().eta())<barrelCutOff_;}
  
};

typedef EBEECutValuesT<double> EBEECutValues;
typedef EBEECutValuesT<int> EBEECutValuesInt;



#endif
