#ifndef JERESProducer_h
#define JERESProducer_h

//
// Author: Sébastien Brochet
//

#include <string>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"

class JetResolutionESProducer : public edm::ESProducer {
private:
  edm::ESGetToken<JME::JetResolutionObject, JetResolutionRcd> token_;

public:
  JetResolutionESProducer(edm::ParameterSet const& fConfig) {
    auto label = fConfig.getParameter<std::string>("label");
    token_ = setWhatProduced(this, label).consumes(edm::ESInputTag{"", label});
  }

  ~JetResolutionESProducer() override {}

  std::unique_ptr<JME::JetResolution> produce(JetResolutionRcd const& iRecord) {
    return std::make_unique<JME::JetResolution>(iRecord.get(token_));
  }
};

class JetResolutionScaleFactorESProducer : public edm::ESProducer {
private:
  edm::ESGetToken<JME::JetResolutionObject, JetResolutionScaleFactorRcd> token_;

public:
  JetResolutionScaleFactorESProducer(edm::ParameterSet const& fConfig) {
    auto label = fConfig.getParameter<std::string>("label");
    token_ = setWhatProduced(this, label).consumes(edm::ESInputTag{"", label});
  }

  ~JetResolutionScaleFactorESProducer() override {}

  std::unique_ptr<JME::JetResolutionScaleFactor> produce(JetResolutionScaleFactorRcd const& iRecord) {
    return std::make_unique<JME::JetResolutionScaleFactor>(iRecord.get(token_));
  }
};
#endif
