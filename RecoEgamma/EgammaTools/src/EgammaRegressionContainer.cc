#include "RecoEgamma/EgammaTools/interface/EgammaRegressionContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"

EgammaRegressionContainer::EgammaRegressionContainer(const edm::ParameterSet& iConfig):
  outputTransformer_(iConfig.getParameter<double>("rangeMin"),iConfig.getParameter<double>("rangeMax")),
  forceHighEnergyTrainingIfSaturated_(iConfig.getParameter<bool>("forceHighEnergyTrainingIfSaturated")),
  lowEtHighEtBoundary_(iConfig.getParameter<double>("lowEtHighEtBoundary")),
  ebLowEtForestName_(iConfig.getParameter<std::string>("ebLowEtForestName")),
  ebHighEtForestName_(iConfig.getParameter<std::string>("ebHighEtForestName")),
  eeLowEtForestName_(iConfig.getParameter<std::string>("eeLowEtForestName")),
  eeHighEtForestName_(iConfig.getParameter<std::string>("eeHighEtForestName")),
  ebLowEtForest_(nullptr),ebHighEtForest_(nullptr),
  eeLowEtForest_(nullptr),eeHighEtForest_(nullptr)
{

}

edm::ParameterSetDescription EgammaRegressionContainer::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<double>("rangeMin",-1.);
  desc.add<double>("rangeMax",3.0);
  desc.add<double>("lowEtHighEtBoundary",50.);
  desc.add<bool>("forceHighEnergyTrainingIfSaturated",false);
  desc.add<std::string>("ebLowEtForestName","electron_eb_ECALTRK_lowpt");
  desc.add<std::string>("ebHighEtForestName","electron_eb_ECALTRK");
  desc.add<std::string>("eeLowEtForestName","electron_ee_ECALTRK_lowpt");
  desc.add<std::string>("eeHighEtForestName","electron_ee_ECALTRK");
  return desc;
}

namespace{
  const GBRForestD* getForest(const edm::EventSetup& iSetup,const std::string&name){
    edm::ESHandle<GBRForestD> handle;
    iSetup.get<GBRDWrapperRcd>().get(name,handle);
    return handle.product();
  }
}

void EgammaRegressionContainer::setEventContent(const edm::EventSetup& iSetup)
{
  ebLowEtForest_  = getForest(iSetup,ebLowEtForestName_);
  ebHighEtForest_ = getForest(iSetup,ebHighEtForestName_);
  eeLowEtForest_  = getForest(iSetup,eeLowEtForestName_);
  eeHighEtForest_ = getForest(iSetup,eeHighEtForestName_); 
}

float EgammaRegressionContainer::operator()(const float et,const bool isEB,const bool isSaturated,const float* data)const
{
  if(useLowEtBin(et,isSaturated)){
    if(isEB) return outputTransformer_(ebLowEtForest_->GetResponse(data));
    else return outputTransformer_(eeLowEtForest_->GetResponse(data));
  }else{
    if(isEB) return outputTransformer_(ebHighEtForest_->GetResponse(data));
    else return outputTransformer_(eeHighEtForest_->GetResponse(data));
  }
}
 
bool EgammaRegressionContainer::useLowEtBin(const float et,const bool isSaturated)const
{
  if(isSaturated && forceHighEnergyTrainingIfSaturated_) return false;
  else return et<lowEtHighEtBoundary_;
}
