// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      Run3ScoutingMETProducer
//
/**\class Run3ScoutingMETProducer Run3ScoutingMETProducer.cc PhysicsTools/PatFromScouting/plugins/Run3ScoutingMETProducer.cc

 Description: Creates pat::MET from scouting MET (pt, phi) stored in event

 Implementation:
     Reads precomputed MET pt and phi from hltScoutingPFPacker and creates pat::MET
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

class Run3ScoutingMETProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingMETProducer(const edm::ParameterSet&);
  ~Run3ScoutingMETProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<double> metPtToken_;
  const edm::EDGetTokenT<double> metPhiToken_;
};

Run3ScoutingMETProducer::Run3ScoutingMETProducer(const edm::ParameterSet& iConfig)
    : metPtToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("metPt"))),
      metPhiToken_(consumes<double>(iConfig.getParameter<edm::InputTag>("metPhi"))) {
  produces<pat::METCollection>();
}

void Run3ScoutingMETProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto patMETs = std::make_unique<pat::METCollection>();

  double metPt = iEvent.get(metPtToken_);
  double metPhi = iEvent.get(metPhiToken_);

  double metPx = metPt * std::cos(metPhi);
  double metPy = metPt * std::sin(metPhi);

  // sumEt is not available in scouting, use metPt as approximation
  double sumEt = metPt;

  reco::MET::LorentzVector p4(metPx, metPy, 0.0, metPt);
  reco::MET::Point vtx(0.0, 0.0, 0.0);

  reco::MET recoMET(sumEt, p4, vtx);
  pat::MET patMET(recoMET);

  // Initialize MET corrections to make NanoAOD happy
  // Using the same values since scouting MET is already the "raw" PF MET
  patMET.setCorShift(metPx, metPy, sumEt, pat::MET::None);
  patMET.setCorShift(metPx, metPy, sumEt, pat::MET::T1);
  patMET.setCorShift(metPx, metPy, sumEt, pat::MET::Calo);
  patMET.setCorShift(metPx, metPy, sumEt, pat::MET::Chs);
  patMET.setCorShift(metPx, metPy, sumEt, pat::MET::Trk);

  patMETs->push_back(patMET);

  iEvent.put(std::move(patMETs));
}

void Run3ScoutingMETProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("metPt", edm::InputTag("hltScoutingPFPacker", "pfMetPt"));
  desc.add<edm::InputTag>("metPhi", edm::InputTag("hltScoutingPFPacker", "pfMetPhi"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingMETProducer);
