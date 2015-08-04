#include "GeneratorInterface/Core/interface/EventVertexHelper.h"
#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"
#include "GeneratorInterface/Core/interface/EventVertexGeneratorFactory.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

EventVertexHelper::EventVertexHelper(edm::ParameterSet const& pset, edm::ConsumesCollector&& iC) :
   vertexGenerator_() {
  edm::ParameterSet const smearingParameters = pset.getParameter<edm::ParameterSet>("VertexSmearing");
  vertexGenerator_ = std::move(edm::EventVertexGeneratorFactory::get()->makeEventVertexGenerator(smearingParameters, iC));
}

EventVertexHelper::~EventVertexHelper() {
}

void
EventVertexHelper::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  vertexGenerator_->beginRun(run, setup);
}

void
EventVertexHelper::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  vertexGenerator_->beginLuminosityBlock(lumi, setup);
}

void
EventVertexHelper::smearVertex(edm::Event const& event, edm::HepMCProduct& product) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(event.streamID());

  vertexGenerator_->generateNewVertex(product, engine);
}
