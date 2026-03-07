// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      PatFromScoutingElectronProducer
//
/**\class PatFromScoutingElectronProducer PatFromScoutingElectronProducer.cc PhysicsTools/PatFromScouting/plugins/PatFromScoutingElectronProducer.cc

 Description: Converts Run3ScoutingElectron to pat::Electron

 Implementation:
     Uses the pat::Electron(const Run3ScoutingElectron&) constructor which sets:
     - Kinematics (pt, eta, phi, mass, charge)
     - Shower shape variables as userFloats (sigmaIetaIeta, hOverE, r9, sMin, sMaj)
     - ID variables as userFloats (dEtaIn, dPhiIn, ooEMOop, missingHits)
     - Track variables as userFloats (trkd0, trkdz, trkpt, trkchi2overndf, trackfbrem)
     - Energy variables as userFloats (rawEnergy, preshowerEnergy, corrEcalEnergyError)
     - Isolation as userFloats (ecalIso, hcalIso, trackIso)
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
    // Constructor now handles kinematics, shower shape, ID, and isolation
    patElectrons->push_back(pat::Electron(sElec));
  }

  iEvent.put(std::move(patElectrons));
}

void PatFromScoutingElectronProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingEgammaPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PatFromScoutingElectronProducer);
