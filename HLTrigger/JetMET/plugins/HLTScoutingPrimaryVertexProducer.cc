// -*- C++ -*-
//
// Package:    HLTrigger/JetMET
// Class:      HLTScoutingPrimaryVertexProducer
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
#include "DataFormats/Math/interface/libminifloat.h"

#include <memory>
#include <utility>

class HLTScoutingPrimaryVertexProducer : public edm::global::EDProducer<> {
public:
  explicit HLTScoutingPrimaryVertexProducer(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const final;

  edm::EDGetTokenT<reco::VertexCollection> const vertexCollToken_;
  int const mantissaPrecision_;
};

HLTScoutingPrimaryVertexProducer::HLTScoutingPrimaryVertexProducer(const edm::ParameterSet& iConfig)
    : vertexCollToken_(consumes(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
      mantissaPrecision_(iConfig.getParameter<int>("mantissaPrecision")) {
  produces<Run3ScoutingVertexCollection>("primaryVtx");
}

void HLTScoutingPrimaryVertexProducer::produce(edm::StreamID sid,
                                               edm::Event& iEvent,
                                               edm::EventSetup const& setup) const {
  auto outVertices = std::make_unique<Run3ScoutingVertexCollection>();

  auto const vertexCollHandle = iEvent.getHandle(vertexCollToken_);

  if (vertexCollHandle.isValid()) {
    outVertices->reserve(vertexCollHandle->size());
    for (auto const& vtx : *vertexCollHandle) {
      outVertices->emplace_back(MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.x(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.y(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.z(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.zError(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.xError(), mantissaPrecision_),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.yError(), mantissaPrecision_),
                                vtx.tracksSize(),
                                MiniFloatConverter::reduceMantissaToNbitsRounding(vtx.chi2(), mantissaPrecision_),
                                vtx.ndof(),
                                vtx.isValid());
    }
  }

  iEvent.put(std::move(outVertices), "primaryVtx");
}

void HLTScoutingPrimaryVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"))
      ->setComment("InputTag of input collection of primary vertices");
  desc.add<int>("mantissaPrecision", 10)->setComment("default float16, change to 23 for float32");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HLTScoutingPrimaryVertexProducer);
