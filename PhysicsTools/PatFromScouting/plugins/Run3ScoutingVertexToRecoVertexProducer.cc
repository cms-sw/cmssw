// -*- C++ -*-
//
// Package:    PhysicsTools/PatFromScouting
// Class:      Run3ScoutingVertexToRecoVertexProducer
//
/**\class Run3ScoutingVertexToRecoVertexProducer Run3ScoutingVertexToRecoVertexProducer.cc PhysicsTools/PatFromScouting/plugins/Run3ScoutingVertexToRecoVertexProducer.cc

 Description: Converts Run3ScoutingVertex to reco::Vertex

 Implementation:
     Uses the existing pat::makeRecoVertex helper function
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

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/PatCandidates/interface/ScoutingDataHandling.h"

class Run3ScoutingVertexToRecoVertexProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingVertexToRecoVertexProducer(const edm::ParameterSet&);
  ~Run3ScoutingVertexToRecoVertexProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<Run3ScoutingVertexCollection> vertexToken_;
};

Run3ScoutingVertexToRecoVertexProducer::Run3ScoutingVertexToRecoVertexProducer(const edm::ParameterSet& iConfig)
    : vertexToken_(consumes<Run3ScoutingVertexCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<reco::VertexCollection>();
}

void Run3ScoutingVertexToRecoVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto recoVertices = std::make_unique<reco::VertexCollection>();

  const auto& scoutingVertices = iEvent.get(vertexToken_);

  for (const auto& sVtx : scoutingVertices) {
    if (!sVtx.isValidVtx()) {
      continue;
    }

    reco::Vertex recoVtx = pat::makeRecoVertex(sVtx);
    recoVertices->push_back(recoVtx);
  }

  iEvent.put(std::move(recoVertices));
}

void Run3ScoutingVertexToRecoVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPrimaryVertexPacker", "primaryVtx"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingVertexToRecoVertexProducer);
