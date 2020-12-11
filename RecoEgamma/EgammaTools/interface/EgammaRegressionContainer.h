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

#include "CondFormats/DataRecord/interface/GBRDWrapperRcd.h"
#include "CondFormats/EgammaObjects/interface/GBRForestD.h"
#include "RecoEgamma/EgammaTools/interface/EgammaBDTOutputTransformer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include <string>

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
  class EventSetup;
  class ConsumesCollector;
}  // namespace edm

class EgammaRegressionContainer {
public:
  EgammaRegressionContainer(const edm::ParameterSet& iConfig, edm::ConsumesCollector& cc);
  ~EgammaRegressionContainer() {}

  static edm::ParameterSetDescription makePSetDescription();

  void setEventContent(const edm::EventSetup& iSetup);

  float operator()(const float et, const bool isEB, const bool isSaturated, const float* data) const;

  bool useLowEtBin(const float et, const bool isSaturated) const;

private:
  const EgammaBDTOutputTransformer outputTransformerLowEt_;
  const EgammaBDTOutputTransformer outputTransformerHighEt_;

  bool forceHighEnergyTrainingIfSaturated_;
  const float lowEtHighEtBoundary_;

  const edm::ESGetToken<GBRForestD, GBRDWrapperRcd> ebLowEtForestToken_;
  const edm::ESGetToken<GBRForestD, GBRDWrapperRcd> ebHighEtForestToken_;
  const edm::ESGetToken<GBRForestD, GBRDWrapperRcd> eeLowEtForestToken_;
  const edm::ESGetToken<GBRForestD, GBRDWrapperRcd> eeHighEtForestToken_;

  const GBRForestD* ebLowEtForest_ = nullptr;   //not owned
  const GBRForestD* ebHighEtForest_ = nullptr;  //not owned
  const GBRForestD* eeLowEtForest_ = nullptr;   //not owned
  const GBRForestD* eeHighEtForest_ = nullptr;  //not owned
};

#endif
