#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/HLTGlobalStatus.h"
#include "DataFormats/Common/interface/HLTPathStatus.h"

#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"

#include <string>

using namespace l1t;

class P2GTTriggerResultsConverter : public edm::stream::EDProducer<> {
public:
  explicit P2GTTriggerResultsConverter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  enum DecisionType { beforeBxMaskAndPrescale, beforePrescale, final };

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<P2GTAlgoBlockMap> algoBlockToken_;
  const std::string prefix_;
  const std::string decisionName_;
  const DecisionType decisionEnum_;

  std::vector<std::string> algoNames_;
};

P2GTTriggerResultsConverter::P2GTTriggerResultsConverter(const edm::ParameterSet& params)
    : algoBlockToken_(consumes<P2GTAlgoBlockMap>(params.getParameter<edm::InputTag>("src"))),
      prefix_(params.getParameter<std::string>("prefix")),
      decisionName_(params.getParameter<std::string>("decision")),
      decisionEnum_(decisionName_ == "beforeBxMaskAndPrescale" ? beforeBxMaskAndPrescale
                    : decisionName_ == "beforePrescale"        ? beforePrescale
                                                               : final) {
  produces<edm::TriggerResults>();
}

void P2GTTriggerResultsConverter::beginRun(const edm::Run&, const edm::EventSetup&) { algoNames_.clear(); }

void P2GTTriggerResultsConverter::produce(edm::Event& event, const edm::EventSetup&) {
  const P2GTAlgoBlockMap& algoBlockMap = event.get(algoBlockToken_);

  edm::HLTGlobalStatus gtDecisions(algoBlockMap.size());

  bool fillAlgoNames = false;

  if (algoNames_.empty()) {
    algoNames_ = std::vector<std::string>(algoBlockMap.size());
    fillAlgoNames = true;
  }

  std::size_t algoIdx = 0;
  for (const auto& [algoName, algoBlock] : algoBlockMap) {
    bool decision = decisionEnum_ == beforeBxMaskAndPrescale ? algoBlock.decisionBeforeBxMaskAndPrescale()
                    : decisionEnum_ == beforePrescale        ? algoBlock.decisionBeforePrescale()
                                                             : algoBlock.decisionFinal();

    gtDecisions[algoIdx] = edm::HLTPathStatus(decision ? edm::hlt::Pass : edm::hlt::Fail);
    if (fillAlgoNames) {
      algoNames_[algoIdx] = prefix_ + algoName + "_" + decisionName_;
    }
    algoIdx++;
  }

  event.put(std::make_unique<edm::TriggerResults>(gtDecisions, algoNames_));
}

void P2GTTriggerResultsConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.add<std::string>("prefix", "L1_");
  desc.ifValue(edm::ParameterDescription<std::string>("decision", "final", true),
               edm::allowedValues<std::string>("beforeBxMaskAndPrescale", "beforePrescale", "final"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(P2GTTriggerResultsConverter);
