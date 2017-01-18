#include "memory"
#include "vector"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FastSimulation/InteractionModel/interface/InteractionModel.h"
#include "FastSimulation/InteractionModel/interface/InteractionModelFactory.h"
#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FastSimulation/Layer/interface/Layer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace fastsim
{
    class SimpleLayerHitProducer : public InteractionModel
    {
    public:
	SimpleLayerHitProducer(const std::string & name,const edm::ParameterSet & cfg);
	void interact(Particle & particle,const Layer & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random) override;
	void registerProducts(edm::ProducerBase & producer) const override;
	void storeProducts(edm::Event & iEvent) override;
    private:
	std::unique_ptr<std::vector<math::XYZTLorentzVector> > layerHits_;
    };
}

fastsim::SimpleLayerHitProducer::SimpleLayerHitProducer(const std::string & name,const edm::ParameterSet & cfg)
    : fastsim::InteractionModel(name)
    , layerHits_(new std::vector<math::XYZTLorentzVector>())
{
}

void fastsim::SimpleLayerHitProducer::registerProducts(edm::ProducerBase & producer) const
{
    LogDebug("FastSimulation") << "      registering products" << std::endl;
    producer.produces<std::vector<math::XYZTLorentzVector> >();
}

void fastsim::SimpleLayerHitProducer::interact(Particle & particle,
					       const fastsim::Layer & layer,
					       std::vector<std::unique_ptr<Particle> > & secondaries,
					       const RandomEngineAndDistribution & random)
{
    if(layer.isOnSurface(particle.position()))
    {
	   layerHits_->push_back(math::XYZTLorentzVector(particle.position().X(),particle.position().Y(),particle.position().Z(),particle.position().T()));
    }
}

void fastsim::SimpleLayerHitProducer::storeProducts(edm::Event & iEvent)
{
    LogDebug("FastSimulation") << "      storing products" << std::endl;
    iEvent.put(std::move(layerHits_));
    //quick fix. Can be done better
    layerHits_.reset(new std::vector<math::XYZTLorentzVector>());
}

DEFINE_EDM_PLUGIN(
    fastsim::InteractionModelFactory,
    fastsim::SimpleLayerHitProducer,
    "fastsim::SimpleLayerHitProducer"
    );
