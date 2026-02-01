// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingElectronProducer
//
/**\class PatFromScoutingElectronProducer PatFromScoutingElectronProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingElectronProducer.cc

 Description: Converts Run3ScoutingElectron to pat::Electron

 Implementation:
     Creates pat::Electron from scouting data, storing shower shape and ID variables as userFloats
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

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"

class PatFromScoutingElectronProducer : public edm::stream::EDProducer<> {
public:
  explicit PatFromScoutingElectronProducer(const edm::ParameterSet&);
  ~PatFromScoutingElectronProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingElectronCollection> electronToken_;
};

PatFromScoutingElectronProducer::PatFromScoutingElectronProducer(const edm::ParameterSet& iConfig)
    : electronToken_(consumes<Run3ScoutingElectronCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<pat::ElectronCollection>();
}

void PatFromScoutingElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patElectrons = std::make_unique<pat::ElectronCollection>();

  const auto& scoutingElectrons = iEvent.get(electronToken_);

  for (const auto& sElec : scoutingElectrons) {
    float px = sElec.pt() * std::cos(sElec.phi());
    float py = sElec.pt() * std::sin(sElec.phi());
    float pz = sElec.pt() * std::sinh(sElec.eta());
    float energy = std::sqrt(px * px + py * py + pz * pz + sElec.m() * sElec.m());

    reco::GsfElectron::LorentzVector p4(px, py, pz, energy);

    int charge = 0;
    if (!sElec.trkcharge().empty()) {
      charge = sElec.trkcharge()[0];
    }

    reco::GsfElectron gsfElec;
    gsfElec.setCharge(charge);
    gsfElec.setP4(p4);
    gsfElec.setVertex(math::XYZPoint(0, 0, 0));

    pat::Electron patElec(gsfElec);

    patElec.addUserFloat("sigmaIetaIeta", sElec.sigmaIetaIeta());
    patElec.addUserFloat("hOverE", sElec.hOverE());
    patElec.addUserFloat("r9", sElec.r9());
    patElec.addUserFloat("sMin", sElec.sMin());
    patElec.addUserFloat("sMaj", sElec.sMaj());

    patElec.addUserFloat("dEtaIn", sElec.dEtaIn());
    patElec.addUserFloat("dPhiIn", sElec.dPhiIn());
    patElec.addUserFloat("ooEMOop", sElec.ooEMOop());
    patElec.addUserInt("missingHits", sElec.missingHits());

    patElec.addUserFloat("trackfbrem", sElec.trackfbrem());
    patElec.addUserFloat("rawEnergy", sElec.rawEnergy());
    patElec.addUserFloat("preshowerEnergy", sElec.preshowerEnergy());
    patElec.addUserFloat("corrEcalEnergyError", sElec.corrEcalEnergyError());

    if (!sElec.trkd0().empty()) {
      patElec.addUserFloat("trkd0", sElec.trkd0()[0]);
    }
    if (!sElec.trkdz().empty()) {
      patElec.addUserFloat("trkdz", sElec.trkdz()[0]);
    }
    if (!sElec.trkpt().empty()) {
      patElec.addUserFloat("trkpt", sElec.trkpt()[0]);
    }
    if (!sElec.trkchi2overndf().empty()) {
      patElec.addUserFloat("trkchi2overndf", sElec.trkchi2overndf()[0]);
    }

    patElec.addUserFloat("ecalIso", sElec.ecalIso());
    patElec.addUserFloat("hcalIso", sElec.hcalIso());
    patElec.addUserFloat("trackIso", sElec.trackIso());

    patElectrons->push_back(patElec);
  }

  iEvent.put(std::move(patElectrons));
}

void PatFromScoutingElectronProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingEgammaPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingElectronProducer);
