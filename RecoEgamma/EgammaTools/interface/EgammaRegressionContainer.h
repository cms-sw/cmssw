#ifndef RecoEgamma_EgammaTools_EgammaRegressionContainer_h
#define RecoEgamma_EgammaTools_EgammaRegressionContainer_h

//author: Sam Harper (RAL)
//description: 
//  egamma energy regressions are binned in barrel/endcap and pt
//  this simply contains the regression for each of (currently) 4 bins
//  as well as the parameters to convert the raw BDT output back to 
//  the physical real value
//  currently e/gamma also can optionally force saturated electrons
//  to always be in the high et training

#include "RecoEgamma/EgammaTools/interface/EgammaBDTOutputTransformer.h"

#include <string>

namespace edm{
  class ParameterSet;
  class ParameterSetDescription;
  class EventSetup;
}
class GBRForestD;

class EgammaRegressionContainer {
public:
  
  EgammaRegressionContainer(const edm::ParameterSet& iConfig);
  ~EgammaRegressionContainer(){}

  static edm::ParameterSetDescription makePSetDescription();

  void setEventContent(const edm::EventSetup& iSetup);  

  float operator()(const float et,const bool isEB,const bool isSaturated,const float* data)const;

private:
  bool useLowEtBin(const float et,const bool isSaturated)const;

  const EgammaBDTOutputTransformer outputTransformer_;
  
  bool forceHighEnergyTrainingIfSaturated_;
  const float lowEtHighEtBoundary_;
  const std::string ebLowEtForestName_;
  const std::string ebHighEtForestName_;
  const std::string eeLowEtForestName_;
  const std::string eeHighEtForestName_;

  const GBRForestD* ebLowEtForest_; //not owned
  const GBRForestD* ebHighEtForest_; //not owned
  const GBRForestD* eeLowEtForest_; //not owned
  const GBRForestD* eeHighEtForest_; //not owned

};


#endif
