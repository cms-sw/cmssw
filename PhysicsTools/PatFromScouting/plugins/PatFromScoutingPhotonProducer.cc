// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingPhotonProducer
//
/**\class PatFromScoutingPhotonProducer PatFromScoutingPhotonProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingPhotonProducer.cc

 Description: Converts Run3ScoutingPhoton to pat::Photon

 Implementation:
     Creates pat::Photon from scouting data, storing shower shape variables as userFloats
*/
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Thu, 05 Dec 2024 15:27:09 GMT
//
//

#include <memory>
#include <cmath>

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
    float px = sPhoton.pt() * std::cos(sPhoton.phi());
    float py = sPhoton.pt() * std::sin(sPhoton.phi());
    float pz = sPhoton.pt() * std::sinh(sPhoton.eta());
    float energy = std::sqrt(px * px + py * py + pz * pz + sPhoton.m() * sPhoton.m());

    reco::Photon::LorentzVector p4(px, py, pz, energy);
    reco::Photon::Point caloPos(0, 0, 0);
    reco::Photon::Point vtx(0, 0, 0);

    reco::Photon recoPhoton(p4, caloPos, reco::PhotonCoreRef(), vtx);
    pat::Photon patPhoton(recoPhoton);

    patPhoton.addUserFloat("sigmaIetaIeta", sPhoton.sigmaIetaIeta());
    patPhoton.addUserFloat("hOverE", sPhoton.hOverE());
    patPhoton.addUserFloat("r9", sPhoton.r9());
    patPhoton.addUserFloat("sMin", sPhoton.sMin());
    patPhoton.addUserFloat("sMaj", sPhoton.sMaj());

    patPhoton.addUserFloat("rawEnergy", sPhoton.rawEnergy());
    patPhoton.addUserFloat("preshowerEnergy", sPhoton.preshowerEnergy());
    patPhoton.addUserFloat("corrEcalEnergyError", sPhoton.corrEcalEnergyError());

    patPhoton.addUserFloat("ecalIso", sPhoton.ecalIso());
    patPhoton.addUserFloat("hcalIso", sPhoton.hcalIso());
    patPhoton.addUserFloat("trkIso", sPhoton.trkIso());

    patPhoton.setIsolation(pat::TrackIso, sPhoton.trkIso());
    patPhoton.setIsolation(pat::EcalIso, sPhoton.ecalIso());
    patPhoton.setIsolation(pat::HcalIso, sPhoton.hcalIso());

    patPhotons->push_back(patPhoton);
  }

  iEvent.put(std::move(patPhotons));
}

void PatFromScoutingPhotonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingEgammaPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingPhotonProducer);
