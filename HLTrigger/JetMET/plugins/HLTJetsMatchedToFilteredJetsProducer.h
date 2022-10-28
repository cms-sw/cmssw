#ifndef HLTrigger_JetMET_HLTJetsMatchedToFilteredJetsProducer_h
#define HLTrigger_JetMET_HLTJetsMatchedToFilteredJetsProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <memory>
#include <utility>

template <typename TriggerJetsType, typename TriggerJetsRefType>
class HLTJetsMatchedToFilteredJetsProducer : public edm::global::EDProducer<> {
public:
  explicit HLTJetsMatchedToFilteredJetsProducer(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  using InputJetCollection = edm::View<TriggerJetsType>;
  using OutputJetCollection = std::vector<TriggerJetsType>;

  // collection of input jets
  edm::EDGetTokenT<InputJetCollection> const jetsToken_;
  // collection of TriggerFilterObjectWithRefs containing TriggerObjects holding refs to Jets stored by an HLTFilter
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> const triggerJetsToken_;
  // TriggerType of Jets produced by previous HLTFilter
  int const triggerJetsType_;
  // maximum Delta-R and Delta-R^2 distances between matched jets
  double const maxDeltaR_, maxDeltaR2_;
};

template <typename TriggerJetsType, typename TriggerJetsRefType>
HLTJetsMatchedToFilteredJetsProducer<TriggerJetsType, TriggerJetsRefType>::HLTJetsMatchedToFilteredJetsProducer(
    edm::ParameterSet const& iConfig)
    : jetsToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      triggerJetsToken_(consumes(iConfig.getParameter<edm::InputTag>("triggerJetsFilter"))),
      triggerJetsType_(iConfig.getParameter<int>("triggerJetsType")),
      maxDeltaR_(iConfig.getParameter<double>("maxDeltaR")),
      maxDeltaR2_(maxDeltaR_ * maxDeltaR_) {
  if (maxDeltaR_ <= 0.) {
    throw cms::Exception("HLTJetsMatchedToFilteredJetsProducerConfiguration")
        << "invalid value for parameter \"maxDeltaR\" (must be > 0): " << maxDeltaR_;
  }

  produces<OutputJetCollection>();
}

template <typename TriggerJetsType, typename TriggerJetsRefType>
void HLTJetsMatchedToFilteredJetsProducer<TriggerJetsType, TriggerJetsRefType>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltJets"));
  desc.add<edm::InputTag>("triggerJetsFilter", edm::InputTag("hltCaloJetsFiltered"));
  desc.add<int>("triggerJetsType", trigger::TriggerJet);
  desc.add<double>("maxDeltaR", 0.5);
  descriptions.addWithDefaultLabel(desc);
}

template <typename TriggerJetsType, typename TriggerJetsRefType>
void HLTJetsMatchedToFilteredJetsProducer<TriggerJetsType, TriggerJetsRefType>::produce(edm::StreamID,
                                                                                        edm::Event& iEvent,
                                                                                        edm::EventSetup const&) const {
  auto const& jets = iEvent.get(jetsToken_);

  std::vector<TriggerJetsRefType> triggerJetsRefVec;
  iEvent.get(triggerJetsToken_).getObjects(triggerJetsType_, triggerJetsRefVec);

  auto outJets = std::make_unique<OutputJetCollection>();
  outJets->reserve(jets.size());

  for (auto const& jet_i : jets) {
    for (auto const& jetRef_j : triggerJetsRefVec) {
      if (reco::deltaR2(jet_i.p4(), jetRef_j->p4()) < maxDeltaR2_) {
        outJets->emplace_back(jet_i);
        break;
      }
    }
  }

  iEvent.put(std::move(outJets));
}

#endif
