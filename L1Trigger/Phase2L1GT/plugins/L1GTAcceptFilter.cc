#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

using namespace l1t;

class L1GTAcceptFilter : public edm::global::EDFilter<> {
public:
  explicit L1GTAcceptFilter(const edm::ParameterSet&);
  ~L1GTAcceptFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  enum DecisionType { beforeBxMaskAndPrescale, beforePrescale, final };

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const edm::EDGetTokenT<P2GTAlgoBlockMap> algoBlocksToken_;
  const DecisionType decisionEnum_;
  int triggerType_;
};

L1GTAcceptFilter::L1GTAcceptFilter(const edm::ParameterSet& config)
    : algoBlocksToken_(consumes<P2GTAlgoBlockMap>(config.getParameter<edm::InputTag>("algoBlocksTag"))),
      decisionEnum_(config.getParameter<std::string>("decision") == "beforeBxMaskAndPrescale" ? beforeBxMaskAndPrescale
                    : config.getParameter<std::string>("decision") == "beforePrescale"        ? beforePrescale
                                                                                              : final),
      triggerType_(config.getParameter<int>("triggerType")) {}

bool L1GTAcceptFilter::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  const P2GTAlgoBlockMap& algoMap = event.get(algoBlocksToken_);
  bool decision = false;
  bool veto = false;
  for (const auto& [name, algoBlock] : algoMap) {
    if (algoBlock.isVeto()) {
      veto |= algoBlock.decisionFinal();
    } else if ((algoBlock.triggerTypes() & triggerType_) > 0) {
      if (decisionEnum_ == beforeBxMaskAndPrescale) {
        decision |= algoBlock.decisionBeforeBxMaskAndPrescale();
      } else if (decisionEnum_ == beforePrescale) {
        decision |= algoBlock.decisionBeforePrescale();
      } else {
        decision |= algoBlock.decisionFinal();
      }
    }
  }

  return decision && !veto;
}

void L1GTAcceptFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("algoBlocksTag");
  desc.add<int>("triggerType", 1);
  desc.ifValue(edm::ParameterDescription<std::string>("decision", "final", true),
               edm::allowedValues<std::string>("beforeBxMaskAndPrescale", "beforePrescale", "final"));

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(L1GTAcceptFilter);
