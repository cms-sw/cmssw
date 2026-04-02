// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingPhotonProducer
//
/**\class PatFromScoutingPhotonProducer PatFromScoutingPhotonProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingPhotonProducer.cc

 Description: Converts Run3ScoutingPhoton to pat::Photon

 Implementation:
     Uses the pat::Photon(const Run3ScoutingPhoton&) constructor which sets:
     - Kinematics (pt, eta, phi, mass)
     - Shower shape variables as userFloats (sigmaIetaIeta, hOverE, r9, sMin, sMaj)
     - Energy variables as userFloats (rawEnergy, preshowerEnergy, corrEcalEnergyError)
     - Isolation as userFloats and PAT isolation keys (ecalIso, hcalIso, trkIso)
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

#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"

class PatFromScoutingPhotonProducer : public edm::stream::EDProducer<> {
public:
  explicit PatFromScoutingPhotonProducer(const edm::ParameterSet&);
  ~PatFromScoutingPhotonProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingPhotonCollection> photonToken_;
};

PatFromScoutingPhotonProducer::PatFromScoutingPhotonProducer(const edm::ParameterSet& iConfig)
    : photonToken_(consumes<Run3ScoutingPhotonCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<pat::PhotonCollection>();
}

void PatFromScoutingPhotonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patPhotons = std::make_unique<pat::PhotonCollection>();

  const auto& scoutingPhotons = iEvent.get(photonToken_);

  for (const auto& sPhoton : scoutingPhotons) {
    // Constructor now handles kinematics, shower shape, and isolation
    patPhotons->push_back(pat::Photon(sPhoton));
  }

  iEvent.put(std::move(patPhotons));
}

void PatFromScoutingPhotonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingEgammaPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingPhotonProducer);
