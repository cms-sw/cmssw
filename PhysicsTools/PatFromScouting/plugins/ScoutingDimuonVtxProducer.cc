// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      ScoutingDimuonVtxProducer
//
/**\class ScoutingDimuonVtxProducer ScoutingDimuonVtxProducer.cc

 Description: Converts scouting displaced dimuon vertices to
 reco::VertexCompositePtrCandidate with CandidatePtr daughters
 pointing into the pat::Muon collection.

 The association between muons and displaced vertices is read from
 Run3ScoutingMuon::vtxIndx(), which stores the indices of the
 displaced vertices each muon belongs to (filled by the HLT
 scouting muon packer).
*/
//
// Original Author:  Dmytro Kovalskyi
//

#include <memory>
#include <vector>
#include <unordered_map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/PatCandidates/interface/ScoutingDataHandling.h"

class ScoutingDimuonVtxProducer : public edm::stream::EDProducer<> {
public:
  explicit ScoutingDimuonVtxProducer(const edm::ParameterSet&);
  ~ScoutingDimuonVtxProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingMuonCollection> scoutMuonToken_;
  const edm::EDGetTokenT<Run3ScoutingVertexCollection> scoutVtxToken_;
  const edm::EDGetTokenT<edm::View<reco::Candidate>> patMuonToken_;
};

ScoutingDimuonVtxProducer::ScoutingDimuonVtxProducer(const edm::ParameterSet& iConfig)
    : scoutMuonToken_(consumes<Run3ScoutingMuonCollection>(iConfig.getParameter<edm::InputTag>("scoutingMuons"))),
      scoutVtxToken_(consumes<Run3ScoutingVertexCollection>(iConfig.getParameter<edm::InputTag>("scoutingVertices"))),
      patMuonToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("patMuons"))) {
  produces<reco::VertexCompositePtrCandidateCollection>();
}

void ScoutingDimuonVtxProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  auto output = std::make_unique<reco::VertexCompositePtrCandidateCollection>();

  const auto& scoutMuons = iEvent.get(scoutMuonToken_);
  const auto& scoutVtxs = iEvent.get(scoutVtxToken_);
  auto patMuonHandle = iEvent.getHandle(patMuonToken_);

  // Build vertex → muon index map by inverting muon.vtxIndx()
  // vtxIndx stores displaced vertex indices that each muon belongs to
  std::unordered_map<int, std::vector<unsigned int>> vtxToMuons;
  for (unsigned int iMu = 0; iMu < scoutMuons.size(); ++iMu) {
    for (int vtxIdx : scoutMuons[iMu].vtxIndx()) {
      vtxToMuons[vtxIdx].push_back(iMu);
    }
  }

  // Convert each scouting displaced vertex
  for (unsigned int iVtx = 0; iVtx < scoutVtxs.size(); ++iVtx) {
    const auto& sVtx = scoutVtxs[iVtx];
    if (!sVtx.isValidVtx())
      continue;

    // Build vertex covariance
    reco::Vertex::Error err;
    err(0, 0) = sVtx.xError() * sVtx.xError();
    err(1, 1) = sVtx.yError() * sVtx.yError();
    err(2, 2) = sVtx.zError() * sVtx.zError();
    err(0, 1) = sVtx.xyCov();
    err(0, 2) = sVtx.xzCov();
    err(1, 2) = sVtx.yzCov();

    // Sum p4 of daughter muons and determine charge
    reco::Candidate::LorentzVector p4;
    int charge = 0;
    auto it = vtxToMuons.find(iVtx);
    if (it != vtxToMuons.end()) {
      for (unsigned int iMu : it->second) {
        if (iMu < patMuonHandle->size()) {
          p4 += patMuonHandle->at(iMu).p4();
          charge += patMuonHandle->at(iMu).charge();
        }
      }
    }

    reco::VertexCompositePtrCandidate vtxCand(
        charge, p4, reco::Candidate::Point(sVtx.x(), sVtx.y(), sVtx.z()), err, sVtx.chi2(), sVtx.ndof());

    // Add daughter CandidatePtrs
    if (it != vtxToMuons.end()) {
      for (unsigned int iMu : it->second) {
        if (iMu < patMuonHandle->size()) {
          vtxCand.addDaughter(patMuonHandle->ptrAt(iMu));
        }
      }
    }

    output->push_back(vtxCand);
  }

  iEvent.put(std::move(output));
}

void ScoutingDimuonVtxProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("scoutingMuons", edm::InputTag("hltScoutingMuonPackerVtx"));
  desc.add<edm::InputTag>("scoutingVertices", edm::InputTag("hltScoutingMuonPackerVtx", "displacedVtx"));
  desc.add<edm::InputTag>("patMuons", edm::InputTag("slimmedMuons"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(ScoutingDimuonVtxProducer);
