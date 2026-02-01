// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      Run3ScoutingMETProducer
//
/**\class Run3ScoutingMETProducer Run3ScoutingMETProducer.cc PhysicsTools/PatFromScouting/plugins/Run3ScoutingMETProducer.cc

 Description: Computes MET from Run3ScoutingParticle collection

 Implementation:
     Computes MET as negative vector sum of all PF candidates
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

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"

class Run3ScoutingMETProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingMETProducer(const edm::ParameterSet&);
  ~Run3ScoutingMETProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingParticleCollection> pfCandToken_;
};

Run3ScoutingMETProducer::Run3ScoutingMETProducer(const edm::ParameterSet& iConfig)
    : pfCandToken_(consumes<Run3ScoutingParticleCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<pat::METCollection>();
}

void Run3ScoutingMETProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patMETs = std::make_unique<pat::METCollection>();

  const auto& pfCandidates = iEvent.get(pfCandToken_);

  double sumPx = 0.0;
  double sumPy = 0.0;
  double sumEt = 0.0;

  for (const auto& pfCand : pfCandidates) {
    double px = pfCand.pt() * std::cos(pfCand.phi());
    double py = pfCand.pt() * std::sin(pfCand.phi());

    sumPx += px;
    sumPy += py;
    sumEt += pfCand.pt();
  }

  double metPx = -sumPx;
  double metPy = -sumPy;
  double metPt = std::sqrt(metPx * metPx + metPy * metPy);

  reco::MET::LorentzVector p4(metPx, metPy, 0.0, metPt);
  reco::MET::Point vtx(0.0, 0.0, 0.0);

  reco::MET recoMET(sumEt, p4, vtx);
  pat::MET patMET(recoMET);

  patMETs->push_back(patMET);

  iEvent.put(std::move(patMETs));
}

void Run3ScoutingMETProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingMETProducer);
