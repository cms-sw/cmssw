// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingMuonProducer
//
/**\class PatFromScoutingMuonProducer PatFromScoutingMuonProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingMuonProducer.cc

 Description: Converts Run3ScoutingMuon to pat::Muon

 Implementation:
     Uses the pat::Muon(const Run3ScoutingMuon&) constructor which sets:
     - Kinematics (pt, eta, phi, mass, charge)
     - Embedded track with vertex
     - Isolation (trackIso, ecalIso, hcalIso -> isolationR03)
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

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"

class PatFromScoutingMuonProducer : public edm::stream::EDProducer<> {
public:
  explicit PatFromScoutingMuonProducer(const edm::ParameterSet&);
  ~PatFromScoutingMuonProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingMuonCollection> muonToken_;
};

PatFromScoutingMuonProducer::PatFromScoutingMuonProducer(const edm::ParameterSet& iConfig)
    : muonToken_(consumes<Run3ScoutingMuonCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<pat::MuonCollection>();
}

void PatFromScoutingMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patMuons = std::make_unique<pat::MuonCollection>();

  const auto& scoutingMuons = iEvent.get(muonToken_);

  for (const auto& sMuon : scoutingMuons) {
    // Constructor now handles kinematics, track embedding, and isolation
    patMuons->push_back(pat::Muon(sMuon));
  }

  iEvent.put(std::move(patMuons));
}

void PatFromScoutingMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingMuonPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingMuonProducer);
