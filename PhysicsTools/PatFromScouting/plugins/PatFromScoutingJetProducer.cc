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
#include "DataFormats/Scouting/interface/Run3ScoutingPFJet.h"

class PatFromScoutingJetProducer : public edm::stream::EDProducer<> {
public:
  explicit PatFromScoutingJetProducer(const edm::ParameterSet&);
  ~PatFromScoutingJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingPFJetCollection> jetToken_;
};

PatFromScoutingJetProducer::PatFromScoutingJetProducer(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<Run3ScoutingPFJetCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<pat::JetCollection>();
}

void PatFromScoutingJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patJets = std::make_unique<pat::JetCollection>();

  const auto& scoutingJets = iEvent.get(jetToken_);

  for (const auto& sJet : scoutingJets) {
    // Constructor now handles kinematics, PFSpecific, and b-tagging
    patJets->push_back(pat::Jet(sJet));
  }

  iEvent.put(std::move(patJets));
}

void PatFromScoutingJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingJetProducer);
