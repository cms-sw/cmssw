// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "SiStripGainFactor.h"

class SiStripGainSimESProducer : public edm::ESProducer {
public:
  SiStripGainSimESProducer(const edm::ParameterSet&);
  ~SiStripGainSimESProducer() override{};

  std::unique_ptr<SiStripGain> produce(const SiStripGainSimRcd&);

private:
  struct TokenLabel {
    TokenLabel(edm::ESConsumesCollector& cc, std::string record, std::string label)
        : token_{cc.consumesFrom<SiStripApvGain, SiStripApvGainSimRcd>(edm::ESInputTag{"", label})},
          recordLabel_{std::move(record), std::move(label)} {}
    edm::ESGetToken<SiStripApvGain, SiStripApvGainSimRcd> token_;
    std::pair<std::string, std::string> recordLabel_;
  };

  std::vector<TokenLabel> tokenLabels_;
  SiStripGainFactor factor_;
};

SiStripGainSimESProducer::SiStripGainSimESProducer(const edm::ParameterSet& iConfig) : factor_{iConfig} {
  auto cc = setWhatProduced(this);

  auto apvGainLabels = iConfig.getParameter<std::vector<edm::ParameterSet> >("APVGain");
  if (apvGainLabels.empty()) {
    throw cms::Exception("Configuration") << "Got empty APVGain vector, but need at least one entry";
  }

  // Fill the vector of apv labels
  for (const auto& gainPSet : apvGainLabels) {
    // Shouldn't all these parameters be tracked?
    tokenLabels_.emplace_back(
        cc, gainPSet.getParameter<std::string>("Record"), gainPSet.getUntrackedParameter<std::string>("Label", ""));
    factor_.push_back_norm(gainPSet.getUntrackedParameter<double>("NormalizationFactor", 1.));
  }

  factor_.resetIfBadNorm();
}

std::unique_ptr<SiStripGain> SiStripGainSimESProducer::produce(const SiStripGainSimRcd& iRecord) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());

  const auto& apvGain = iRecord.get(tokenLabels_[0].token_);
  auto gain = std::make_unique<SiStripGain>(apvGain, factor_.get(apvGain, 0), tokenLabels_[0].recordLabel_, detInfo);

  for (unsigned int i = 1; i < tokenLabels_.size(); ++i) {
    const auto& apvGain = iRecord.get(tokenLabels_[i].token_);
    gain->multiply(apvGain, factor_.get(apvGain, i), tokenLabels_[i].recordLabel_, detInfo);
  }
  return gain;
}

#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiStripGainSimESProducer);
