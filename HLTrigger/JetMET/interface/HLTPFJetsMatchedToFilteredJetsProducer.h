#ifndef HLTrigger_JetMET_HLTPFJetsMatchedToFilteredJetsProducer_h
#define HLTrigger_JetMET_HLTPFJetsMatchedToFilteredJetsProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>
#include <memory>
#include <utility>

template <typename TriggerJetsRefType>
class HLTPFJetsMatchedToFilteredJetsProducer : public edm::global::EDProducer<> {
public:
  explicit HLTPFJetsMatchedToFilteredJetsProducer(edm::ParameterSet const& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  // collection of reco::*Jets (using edm::View<reco::Candidate> to be able to read different types of reco::*Jet collections without templating explicitly)
  edm::EDGetTokenT<edm::View<reco::Candidate>> const recoCandsToken_;
  // collection of TriggerFilterObjectWithRefs containing TriggerObjects holding refs to Jets stored by an HLTFilter
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> const triggerJetsToken_;
  // TriggerType of Jets produced by previous HLTFilter
  int const triggerJetsType_;
  // maximum Delta-R and Delta-R^2 distances between matched jets
  double const maxDeltaR_, maxDeltaR2_;
};

template <typename TriggerJetsRefType>
HLTPFJetsMatchedToFilteredJetsProducer<TriggerJetsRefType>::HLTPFJetsMatchedToFilteredJetsProducer(
    edm::ParameterSet const& iConfig)
    : recoCandsToken_(consumes(iConfig.getParameter<edm::InputTag>("src"))),
      triggerJetsToken_(consumes(iConfig.getParameter<edm::InputTag>("triggerJetsFilter"))),
      triggerJetsType_(iConfig.getParameter<int>("triggerJetsType")),
      maxDeltaR_(iConfig.getParameter<double>("maxDeltaR")),
      maxDeltaR2_(maxDeltaR_ * maxDeltaR_) {
  if (maxDeltaR_ <= 0.) {
    throw cms::Exception("HLTPFJetsMatchedToFilteredJetsProducerConfiguration")
        << "invalid value for parameter \"DeltaR\" (must be > 0): " << maxDeltaR_;
  }

  produces<reco::PFJetCollection>();
}

template <typename TriggerJetsRefType>
void HLTPFJetsMatchedToFilteredJetsProducer<TriggerJetsRefType>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltPFJets"));
  desc.add<edm::InputTag>("triggerJetsFilter", edm::InputTag("hltSingleJet240Regional"));
  desc.add<int>("triggerJetsType", trigger::TriggerJet);
  desc.add<double>("maxDeltaR", 0.5);
  descriptions.addWithDefaultLabel(desc);
}

template <typename TriggerJetsRefType>
void HLTPFJetsMatchedToFilteredJetsProducer<TriggerJetsRefType>::produce(edm::StreamID streamID,
                                                                         edm::Event& iEvent,
                                                                         edm::EventSetup const&) const {
  auto const& recoCands = iEvent.get(recoCandsToken_);

  std::vector<TriggerJetsRefType> triggerJetsRefVec;
  auto const& triggerJets = iEvent.get(triggerJetsToken_);
  triggerJets.getObjects(triggerJetsType_, triggerJetsRefVec);

  math::XYZPoint const pvtxPoint(0., 0., 0.);
  reco::PFJet::Specific const pfJetSpec;

  auto outPFJets = std::make_unique<reco::PFJetCollection>();
  outPFJets->reserve(recoCands.size());

  for (auto const& iJetRef : triggerJetsRefVec) {
    for (auto const& jRecoCand : recoCands) {
      auto const dR2 = reco::deltaR2(jRecoCand.p4(), iJetRef->p4());
      if (dR2 < maxDeltaR2_) {
        outPFJets->emplace_back(jRecoCand.p4(), pvtxPoint, pfJetSpec);
        break;
      }
    }
  }

  iEvent.put(std::move(outPFJets));
}

#endif
