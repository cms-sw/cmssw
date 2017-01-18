#include "memory"
#include "vector"
#include "FastSimulation/InteractionModel/interface/InteractionModel.h"
#include "FastSimulation/InteractionModel/interface/InteractionModelFactory.h"
#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FastSimulation/Layer/interface/Layer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace fastsim
{
    class DummyHitProducer : public InteractionModel
    {
    public:
	DummyHitProducer(const std::string & name,const edm::ParameterSet & cfg);
	void interact(Particle & particle,const Layer & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random) override;
	void registerProducts(edm::ProducerBase & producer) const override;
	void storeProducts(edm::Event & iEvent) override;
    };
}

fastsim::DummyHitProducer::DummyHitProducer(const std::string & name,const edm::ParameterSet & cfg)
    : fastsim::InteractionModel(name)
{
}

void fastsim::DummyHitProducer::registerProducts(edm::ProducerBase & producer) const
{
    LogDebug("FastSimulation") << "      registering products" << std::endl;
    producer.produces<edm::PSimHitContainer>("MuonCSCHits");
    producer.produces<edm::PSimHitContainer>("MuonDTHits");
    producer.produces<edm::PSimHitContainer>("MuonRPCHits");
    producer.produces<edm::PCaloHitContainer>("EcalHitsEB");
    producer.produces<edm::PCaloHitContainer>("EcalHitsEE");
    producer.produces<edm::PCaloHitContainer>("EcalHitsES");
    producer.produces<edm::PCaloHitContainer>("HcalHits");
}

void fastsim::DummyHitProducer::interact(Particle & particle,
					       const fastsim::Layer & layer,
					       std::vector<std::unique_ptr<Particle> > & secondaries,
					       const RandomEngineAndDistribution & random)
{
    // I'm a dummy...
    return;
}

void fastsim::DummyHitProducer::storeProducts(edm::Event & iEvent)
{
    LogDebug("FastSimulation") << "      storing products" << std::endl;
    iEvent.put(std::move(std::unique_ptr<edm::PSimHitContainer>(new edm::PSimHitContainer())),"MuonCSCHits");
    iEvent.put(std::move(std::unique_ptr<edm::PSimHitContainer>(new edm::PSimHitContainer())),"MuonDTHits");
    iEvent.put(std::move(std::unique_ptr<edm::PSimHitContainer>(new edm::PSimHitContainer())),"MuonRPCHits");
    iEvent.put(std::move(std::unique_ptr<edm::PCaloHitContainer>(new edm::PCaloHitContainer())),"EcalHitsEB");
    iEvent.put(std::move(std::unique_ptr<edm::PCaloHitContainer>(new edm::PCaloHitContainer())),"EcalHitsEE");
    iEvent.put(std::move(std::unique_ptr<edm::PCaloHitContainer>(new edm::PCaloHitContainer())),"EcalHitsES");
    iEvent.put(std::move(std::unique_ptr<edm::PCaloHitContainer>(new edm::PCaloHitContainer())),"HcalHits");
}

DEFINE_EDM_PLUGIN(
    fastsim::InteractionModelFactory,
    fastsim::DummyHitProducer,
    "fastsim::DummyHitProducer"
    );
