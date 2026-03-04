// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingJetProducer
//
/**\class PatFromScoutingJetProducer PatFromScoutingJetProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingJetProducer.cc

 Description: Converts Run3ScoutingPFJet to pat::Jet

 Implementation:
     Uses the pat::Jet(const Run3ScoutingPFJet&) constructor which sets:
     - Kinematics (pt, eta, phi, mass)
     - Jet area
     - PFJet::Specific (energy fractions, multiplicities)
     - B-tagging discriminators (CSV, DeepCSV)

     Constituent indices from the scouting jet are resolved to
     CandidatePtr daughters pointing into the PackedCandidateCollection.
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Thu, 05 Dec 2024 15:27:09 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"

class PatFromScoutingJetProducer : public edm::stream::EDProducer<> {
public:
  explicit PatFromScoutingJetProducer(const edm::ParameterSet&);
  ~PatFromScoutingJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingPFJetCollection> jetToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandToken_;
};

PatFromScoutingJetProducer::PatFromScoutingJetProducer(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<Run3ScoutingPFJetCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      pfCandToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidates"))) {
  produces<pat::JetCollection>();
}

void PatFromScoutingJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patJets = std::make_unique<pat::JetCollection>();

  const auto& scoutingJets = iEvent.get(jetToken_);
  auto pfCandHandle = iEvent.getHandle(pfCandToken_);

  for (const auto& sJet : scoutingJets) {
    pat::Jet jet(sJet);

    // Resolve constituent indices to CandidatePtr daughters
    for (int idx : sJet.constituents()) {
      if (idx >= 0 && idx < static_cast<int>(pfCandHandle->size())) {
        reco::CandidatePtr ptr(pfCandHandle, idx);
        jet.addDaughter(ptr);
      }
    }

    patJets->push_back(jet);
  }

  iEvent.put(std::move(patJets));
}

void PatFromScoutingJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker"));
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingJetProducer);
