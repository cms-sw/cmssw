// -*- C++ -*-
//
// Package:    HLTrigger/JetMET
// Class:      HLTScoutingPrimaryVertexProducer

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/Math/interface/deltaR.h"

class HLTScoutingPrimaryVertexProducer : public edm::global::EDProducer<> {
public:
  explicit HLTScoutingPrimaryVertexProducer(const edm::ParameterSet&);
  ~HLTScoutingPrimaryVertexProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const final;
  const edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
};

//
// constructors and destructor
//
HLTScoutingPrimaryVertexProducer::HLTScoutingPrimaryVertexProducer(const edm::ParameterSet& iConfig)
    : vertexCollection_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"))) {
  //register products
  produces<Run3ScoutingVertexCollection>("primaryVtx");
}

HLTScoutingPrimaryVertexProducer::~HLTScoutingPrimaryVertexProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingPrimaryVertexProducer::produce(edm::StreamID sid,
                                               edm::Event& iEvent,
                                               edm::EventSetup const& setup) const {
  using namespace edm;

  //get vertices
  Handle<reco::VertexCollection> vertexCollection;

  std::unique_ptr<Run3ScoutingVertexCollection> outVertices(new Run3ScoutingVertexCollection());

  if (iEvent.getByToken(vertexCollection_, vertexCollection)) {
    for (auto& vtx : *vertexCollection) {
      outVertices->emplace_back(vtx.x(),
                                vtx.y(),
                                vtx.z(),
                                vtx.zError(),
                                vtx.xError(),
                                vtx.yError(),
                                vtx.tracksSize(),
                                vtx.chi2(),
                                vtx.ndof(),
                                vtx.isValid());
    }
  }

  //put output
  iEvent.put(std::move(outVertices), "primaryVtx");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingPrimaryVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"));
  descriptions.add("hltScoutingPrimaryVertexProducer", desc);
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTScoutingPrimaryVertexProducer);
