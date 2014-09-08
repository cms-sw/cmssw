#ifndef EgammaAnalysis_ElectronTools_EBEECutValues_h
#define EgammaAnalysis_ElectronTools_EBEECutValues_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

namespace reco {
  typedef edm::Ptr<reco::GsfElectron> GsfElectronPtr;
}

class EBEECutValues {
private:
  double barrel_;
  double endcap_;
  const double barrelCutOff_=1.479; //this is currrently used to identify if object is barrel or endcap but may change
  
public:
  EBEECutValues(const edm::ParameterSet& params,const std::string& name):
    barrel_(params.getParameter<double>(name+"EB")),
    endcap_(params.getParameter<double>(name+"EE")){}
  double operator()(const reco::GsfElectronPtr& cand)const{return isBarrel(cand) ? barrel_ : endcap_;}

private:
  const bool isBarrel(const reco::GsfElectronPtr& cand)const{return std::abs(cand->superCluster()->position().eta())<barrelCutOff_;}
  
};

#endif
