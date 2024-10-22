#include <cmath>
#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "HLTrigger/JetMET/interface/HLTJetL1TMatchProducer.h"

template <typename T>
HLTJetL1TMatchProducer<T>::HLTJetL1TMatchProducer(const edm::ParameterSet& iConfig) {
  jetsInput_ = iConfig.template getParameter<edm::InputTag>("jetsInput");
  L1Jets_ = iConfig.template getParameter<edm::InputTag>("L1Jets");

  // minimum delta-R^2 threshold with sign
  auto const DeltaR = iConfig.template getParameter<double>("DeltaR");
  DeltaR2_ = DeltaR * std::abs(DeltaR);

  m_theJetToken = consumes(jetsInput_);
  m_theL1JetToken = consumes(L1Jets_);

  produces<std::vector<T>>();
}

template <typename T>
void HLTJetL1TMatchProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsInput", edm::InputTag("hltAntiKT5PFJets"));
  desc.add<edm::InputTag>("L1Jets", edm::InputTag("hltCaloStage2Digis"));
  desc.add<double>("DeltaR", 0.5);
  descriptions.add(defaultModuleLabel<HLTJetL1TMatchProducer<T>>(), desc);
}

template <typename T>
void HLTJetL1TMatchProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& jets = iEvent.get(m_theJetToken);
  auto const l1Jets = iEvent.getHandle(m_theL1JetToken);

  bool trigger_bx_only = true;  // selection of BX not implemented

  auto result = std::make_unique<std::vector<T>>();

  if (l1Jets.isValid()) {
    for (auto const& jet : jets) {
      bool isMatched = false;
      for (int ibx = l1Jets->getFirstBX(); ibx <= l1Jets->getLastBX(); ++ibx) {
        if (trigger_bx_only && (ibx != 0))
          continue;
        for (auto it = l1Jets->begin(ibx); it != l1Jets->end(ibx); it++) {
          if (it->et() == 0)
            continue;  // if you don't care about L1T candidates with zero ET.
          if (reco::deltaR2(jet.eta(), jet.phi(), it->eta(), it->phi()) < DeltaR2_) {
            isMatched = true;
            break;
          }
        }
      }
      if (isMatched) {
        result->emplace_back(jet);
      }
    }
  } else {
    edm::LogWarning("MissingProduct") << "L1Upgrade l1Jets bx collection not found.";
  }

  iEvent.put(std::move(result));
}
