#include "IOMC/EventVertexGenerators/interface/EventVertexProducer.h"
#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"
#include "GeneratorInterface/Core/interface/EventVertexGeneratorFactory.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

EventVertexProducer::EventVertexProducer(edm::ParameterSet const& pset) :
   sourceToken_(consumes<edm::HepMCProduct>(pset.getParameter<edm::InputTag>("src"))),
   vertexGenerator_() {
  edm::ConsumesCollector iC = consumesCollector();  
  edm::ParameterSet const smearingParameters = pset.getParameter<edm::ParameterSet>("VertexSmearing");
  vertexGenerator_ = std::move(edm::EventVertexGeneratorFactory::get()->makeEventVertexGenerator(smearingParameters, iC));
  produces<edm::HepMCProduct>();
}

EventVertexProducer::~EventVertexProducer() {
}

void
EventVertexProducer::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  vertexGenerator_->beginRun(run, setup);
}

void
EventVertexProducer::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  vertexGenerator_->beginLuminosityBlock(lumi, setup);
}

void
EventVertexProducer::produce(edm::Event& event, edm::EventSetup const& setup) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(event.streamID());

  edm::Handle<edm::HepMCProduct> HepMCEvt;
  event.getByToken(sourceToken_, HepMCEvt);

  std::unique_ptr<edm::HepMCProduct> product(new edm::HepMCProduct(*HepMCEvt));

  vertexGenerator_->generateNewVertex(*product, engine);

  event.put(std::move(product));
}

