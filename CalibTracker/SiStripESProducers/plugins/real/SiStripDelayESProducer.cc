// -*- C++ -*-
//
// Package:    SiStripDelayESProducer
// Class:      SiStripDelayESProducer
//
/**\class SiStripDelayESProducer SiStripDelayESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripDelayESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  M. De Mattia
//         Created:  26/10/2010
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDelay.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"

class SiStripDelayESProducer : public edm::ESProducer {
public:
  SiStripDelayESProducer(const edm::ParameterSet&);
  ~SiStripDelayESProducer() override{};

  std::unique_ptr<SiStripDelay> produce(const SiStripDelayRcd&);

private:
  struct TokenSign {
    TokenSign(edm::ESConsumesCollector& cc, const std::string& recordName, const std::string& label, int sumSign)
        : token_{cc.consumesFrom<SiStripBaseDelay, SiStripBaseDelayRcd>(edm::ESInputTag{"", label})},
          recordName_{recordName},
          label_{label},
          sumSign_{sumSign} {}
    edm::ESGetToken<SiStripBaseDelay, SiStripBaseDelayRcd> token_;
    std::string recordName_;
    std::string label_;
    int sumSign_;
  };
  std::vector<TokenSign> tokens_;
};

SiStripDelayESProducer::SiStripDelayESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);

  edm::LogInfo("SiStripDelayESProducer") << "ctor" << std::endl;

  for (const auto& pset : iConfig.getParameter<std::vector<edm::ParameterSet>>("ListOfRecordToMerge")) {
    auto recordName = pset.getParameter<std::string>("Record");
    auto label = pset.getParameter<std::string>("Label");

    edm::LogInfo("SiStripDelayESProducer")
        << "[SiStripDelayESProducer::ctor] Going to get data from record " << recordName << " with label " << label;

    // Is the "recordName" parameter really useful?
    if (recordName == "SiStripBaseDelayRcd") {
      tokens_.emplace_back(cc, recordName, label, pset.getParameter<int>("SumSign"));
    } else {
      // Would an exception make sense?
      edm::LogError("SiStripDelayESProducer")
          << "[SiStripDelayESProducer::ctor] Skipping the requested data for unexisting record " << recordName
          << " with tag " << label << std::endl;
    }
  }
}

std::unique_ptr<SiStripDelay> SiStripDelayESProducer::produce(const SiStripDelayRcd& iRecord) {
  edm::LogInfo("SiStripDelayESProducer") << "produce called" << std::endl;
  auto delay = std::make_unique<SiStripDelay>();

  for (const auto& tokenSign : tokens_) {
    const auto& baseDelay = iRecord.get(tokenSign.token_);
    delay->fillNewDelay(baseDelay, tokenSign.sumSign_, std::make_pair(tokenSign.recordName_, tokenSign.label_));
  }

  delay->makeDelay();

  return delay;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripDelayESProducer);
