#include "RecoEgamma/EgammaTools/interface/EgammaRegressionContainer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

EgammaRegressionContainer::EgammaRegressionContainer(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc)
    : outputTransformerLowEt_(iConfig.getParameter<double>("rangeMinLowEt"),
                              iConfig.getParameter<double>("rangeMaxLowEt")),
      outputTransformerHighEt_(iConfig.getParameter<double>("rangeMinHighEt"),
                               iConfig.getParameter<double>("rangeMaxHighEt")),
      forceHighEnergyTrainingIfSaturated_(iConfig.getParameter<bool>("forceHighEnergyTrainingIfSaturated")),
      lowEtHighEtBoundary_(iConfig.getParameter<double>("lowEtHighEtBoundary")),
      ebLowEtForestToken_{cc.esConsumes(iConfig.getParameter<edm::ESInputTag>("ebLowEtForestName"))},
      ebHighEtForestToken_{cc.esConsumes(iConfig.getParameter<edm::ESInputTag>("ebHighEtForestName"))},
      eeLowEtForestToken_{cc.esConsumes(iConfig.getParameter<edm::ESInputTag>("eeLowEtForestName"))},
      eeHighEtForestToken_{cc.esConsumes(iConfig.getParameter<edm::ESInputTag>("eeHighEtForestName"))} {}

edm::ParameterSetDescription EgammaRegressionContainer::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.add<double>("rangeMinLowEt", -1.);
  desc.add<double>("rangeMaxLowEt", 3.0);
  desc.add<double>("rangeMinHighEt", -1.);
  desc.add<double>("rangeMaxHighEt", 3.0);
  desc.add<double>("lowEtHighEtBoundary", 50.);
  desc.add<bool>("forceHighEnergyTrainingIfSaturated", false);
  desc.add<edm::ESInputTag>("ebLowEtForestName", edm::ESInputTag{"", "electron_eb_ECALTRK_lowpt"});
  desc.add<edm::ESInputTag>("ebHighEtForestName", edm::ESInputTag{"", "electron_eb_ECALTRK"});
  desc.add<edm::ESInputTag>("eeLowEtForestName", edm::ESInputTag{"", "electron_ee_ECALTRK_lowpt"});
  desc.add<edm::ESInputTag>("eeHighEtForestName", edm::ESInputTag{"", "electron_ee_ECALTRK"});
  return desc;
}

void EgammaRegressionContainer::setEventContent(const edm::EventSetup& iSetup) {
  ebLowEtForest_ = &iSetup.getData(ebLowEtForestToken_);
  ebHighEtForest_ = &iSetup.getData(ebHighEtForestToken_);
  eeLowEtForest_ = &iSetup.getData(eeLowEtForestToken_);
  eeHighEtForest_ = &iSetup.getData(eeHighEtForestToken_);
}

float EgammaRegressionContainer::operator()(const float et,
                                            const bool isEB,
                                            const bool isSaturated,
                                            const float* data) const {
  if (useLowEtBin(et, isSaturated)) {
    if (isEB)
      return outputTransformerLowEt_(ebLowEtForest_->GetResponse(data));
    else
      return outputTransformerLowEt_(eeLowEtForest_->GetResponse(data));
  } else {
    if (isEB)
      return outputTransformerHighEt_(ebHighEtForest_->GetResponse(data));
    else
      return outputTransformerHighEt_(eeHighEtForest_->GetResponse(data));
  }
}

bool EgammaRegressionContainer::useLowEtBin(const float et, const bool isSaturated) const {
  if (isSaturated && forceHighEnergyTrainingIfSaturated_)
    return false;
  else
    return et < lowEtHighEtBoundary_;
}
