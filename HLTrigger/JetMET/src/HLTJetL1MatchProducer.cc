#include <cmath>
#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "HLTrigger/JetMET/interface/HLTJetL1MatchProducer.h"

template <typename T>
HLTJetL1MatchProducer<T>::HLTJetL1MatchProducer(const edm::ParameterSet& iConfig) {
  jetsInput_ = iConfig.template getParameter<edm::InputTag>("jetsInput");
  L1TauJets_ = iConfig.template getParameter<edm::InputTag>("L1TauJets");
  L1CenJets_ = iConfig.template getParameter<edm::InputTag>("L1CenJets");
  L1ForJets_ = iConfig.template getParameter<edm::InputTag>("L1ForJets");

  // minimum delta-R^2 threshold with sign
  auto const DeltaR = iConfig.template getParameter<double>("DeltaR");
  DeltaR2_ = DeltaR * std::abs(DeltaR);

  typedef std::vector<T> TCollection;
  m_theJetToken = consumes<TCollection>(jetsInput_);
  m_theL1TauJetToken = consumes<l1extra::L1JetParticleCollection>(L1TauJets_);
  m_theL1CenJetToken = consumes<l1extra::L1JetParticleCollection>(L1CenJets_);
  m_theL1ForJetToken = consumes<l1extra::L1JetParticleCollection>(L1ForJets_);
  produces<TCollection>();
}

template <typename T>
void HLTJetL1MatchProducer<T>::beginJob() {}

template <typename T>
HLTJetL1MatchProducer<T>::~HLTJetL1MatchProducer() = default;

template <typename T>
void HLTJetL1MatchProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsInput", edm::InputTag("hltAntiKT5PFJets"));
  desc.add<edm::InputTag>("L1TauJets", edm::InputTag("hltL1extraParticles", "Tau"));
  desc.add<edm::InputTag>("L1CenJets", edm::InputTag("hltL1extraParticles", "Central"));
  desc.add<edm::InputTag>("L1ForJets", edm::InputTag("hltL1extraParticles", "Forward"));
  desc.add<double>("DeltaR", 0.5);
  descriptions.add(defaultModuleLabel<HLTJetL1MatchProducer<T>>(), desc);
}

template <typename T>
void HLTJetL1MatchProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& jets = iEvent.get(m_theJetToken);

  auto result = std::make_unique<std::vector<T>>();
  result->reserve(jets.size());

  auto const& l1TauJets = iEvent.get(m_theL1TauJetToken);
  auto const& l1CenJets = iEvent.get(m_theL1CenJetToken);
  auto const& l1ForJets = iEvent.get(m_theL1ForJetToken);

  for (auto const& jet : jets) {
    bool isMatched = false;

    for (auto const& l1t_obj : l1TauJets) {
      if (reco::deltaR2(jet.eta(), jet.phi(), l1t_obj.eta(), l1t_obj.phi()) < DeltaR2_) {
        isMatched = true;
        break;
      }
    }

    if (isMatched) {
      result->emplace_back(jet);
      continue;
    }

    for (auto const& l1t_obj : l1CenJets) {
      if (reco::deltaR2(jet.eta(), jet.phi(), l1t_obj.eta(), l1t_obj.phi()) < DeltaR2_) {
        isMatched = true;
        break;
      }
    }

    if (isMatched) {
      result->emplace_back(jet);
      continue;
    }

    for (auto const& l1t_obj : l1ForJets) {
      if (reco::deltaR2(jet.eta(), jet.phi(), l1t_obj.eta(), l1t_obj.phi()) < DeltaR2_) {
        isMatched = true;
        break;
      }
    }

    if (isMatched) {
      result->emplace_back(jet);
      continue;
    }
  }

  iEvent.put(std::move(result));
}
