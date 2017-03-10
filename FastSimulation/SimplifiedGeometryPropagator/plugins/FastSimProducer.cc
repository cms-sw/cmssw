// system include files
#include <memory>
#include <string>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

// fastsim
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Geometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Decayer.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/LayerNavigator.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleFilter.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"  // TODO: get rid of this
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModel.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/InteractionModelFactory.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleManager.h"

// other

class FastSimProducer : public edm::stream::EDProducer<> {
public:

    explicit FastSimProducer(const edm::ParameterSet&);
    ~FastSimProducer(){;}

private:

	virtual void beginStream(edm::StreamID id);
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endStream();

    edm::EDGetTokenT<edm::HepMCProduct> genParticlesToken_;
    fastsim::Geometry geometry_;
    double beamPipeRadius_;
    fastsim::ParticleFilter particleFilter_;
    std::unique_ptr<RandomEngineAndDistribution> _randomEngine;
    fastsim::Decayer decayer_;
    std::vector<std::unique_ptr<fastsim::InteractionModel> > interactionModels_;
    std::map<std::string,fastsim::InteractionModel *> interactionModelMap_;
    edm::IOVSyncValue iovSyncValue_;
    static const std::string MESSAGECATEGORY;
};

const std::string FastSimProducer::MESSAGECATEGORY = "FastSimulation";

FastSimProducer::FastSimProducer(const edm::ParameterSet& iConfig)
    : genParticlesToken_(consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("src"))) 
    , geometry_(iConfig.getParameter<edm::ParameterSet>("detectorDefinition"))
    , beamPipeRadius_(iConfig.getParameter<double>("beamPipeRadius"))
    , particleFilter_(iConfig.getParameter<edm::ParameterSet>("particleFilter"))
    , _randomEngine(nullptr)
{

    //----------------
    // define interaction models
    //---------------
    const edm::ParameterSet & modelCfgs = iConfig.getParameter<edm::ParameterSet>("interactionModels");
    for( const std::string & modelName : modelCfgs.getParameterNames())
    {
		const edm::ParameterSet & modelCfg = modelCfgs.getParameter<edm::ParameterSet>(modelName);
		std::string modelClassName(modelCfg.getParameter<std::string>("className"));
		std::unique_ptr<fastsim::InteractionModel> interactionModel(fastsim::InteractionModelFactory::get()->create(modelClassName,modelName,modelCfg));
		if(!interactionModel.get()){
			throw cms::Exception("FastSimProducer") << "InteractionModel " << modelName << " could not be created" << std::endl;
		}
		interactionModels_.push_back(std::move(interactionModel));
		interactionModelMap_[modelName] = interactionModels_.back().get();
    }

    //----------------
    // register products
    //----------------
    produces<edm::SimTrackContainer>();
    produces<edm::SimVertexContainer>();
    for(auto & interactionModel : interactionModels_)
    {
		interactionModel->registerProducts(*this);
    }
}

void
FastSimProducer::beginStream(const edm::StreamID id)
{
    _randomEngine = std::make_unique<RandomEngineAndDistribution>(id);
}

void
FastSimProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    LogDebug(MESSAGECATEGORY) << "   produce";

    // do the iov thing
    if(iovSyncValue_!=iSetup.iovSyncValue())
    {
		LogDebug(MESSAGECATEGORY) << "   triggering update of event setup" << std::endl;
		iovSyncValue_=iSetup.iovSyncValue();
		geometry_.update(iSetup,interactionModelMap_);
    }

    std::unique_ptr<edm::SimTrackContainer> output_simTracks(new edm::SimTrackContainer);
    std::unique_ptr<edm::SimVertexContainer> output_simVertices(new edm::SimVertexContainer);

    edm::ESHandle < HepPDT::ParticleDataTable > pdt;
    iSetup.getData(pdt);
    // TODO: get rid of this
    ParticleTable::Sentry ptable(&(*pdt));

    edm::Handle<edm::HepMCProduct> genParticles;
    iEvent.getByToken(genParticlesToken_,genParticles);

    fastsim::ParticleManager particleManager(
	*genParticles->GetEvent()
	,*pdt
	,beamPipeRadius_
	,particleFilter_
	,output_simTracks
	,output_simVertices);
	
    // loop over particles
    LogDebug(MESSAGECATEGORY) << "################################"
			      << "\n###############################";    

    for(std::unique_ptr<fastsim::Particle> particle = particleManager.nextParticle(*_randomEngine); particle != 0;particle=particleManager.nextParticle(*_randomEngine)) 
    {
    	LogDebug(MESSAGECATEGORY) << "\n   moving NEXT particle: " << *particle;

		// move the particle through the layers
		fastsim::LayerNavigator layerNavigator(geometry_);
		const fastsim::SimplifiedGeometry * layer = 0;
		while(layerNavigator.moveParticleToNextLayer(*particle,layer))
		{
		    if(layer) LogDebug(MESSAGECATEGORY) << "   moved to next layer: " << *layer;
			LogDebug(MESSAGECATEGORY) <<  "   new state: " << *particle;

			// do decays
			if(!particle->isStable() && particle->remainingProperLifeTime() < 1E-20)
			{
			    LogDebug(MESSAGECATEGORY) << "Decaying particle...";
			    std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
			    decayer_.decay(*particle,secondaries,_randomEngine->theEngine());
			    LogDebug(MESSAGECATEGORY) << "   decay has " << secondaries.size() << " products";
			    particleManager.addSecondaries(particle->position(),particle->simTrackIndex(),secondaries);
			    break;
			}

		    
		    // perform interaction between layer and particle
		    for(fastsim::InteractionModel * interactionModel : layer->getInteractionModels())
		    {
				LogDebug(MESSAGECATEGORY) << "   interact with " << *interactionModel;
				std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
				interactionModel->interact(*particle,*layer,secondaries,*_randomEngine);
				particleManager.addSecondaries(particle->position(),particle->simTrackIndex(),secondaries);
		    }

		    // kinematic cuts
		    // temporary: break after 100 ns
		    if(particle->position().T() > 100)
		    {
				break;
		    }
		    
		    LogDebug(MESSAGECATEGORY) << "--------------------------------"
					      << "\n-------------------------------";

		}
		
		LogDebug(MESSAGECATEGORY) << "################################"
					  << "\n###############################";
    }

    // store simHits and simTracks
    iEvent.put(particleManager.harvestSimTracks());
    iEvent.put(particleManager.harvestSimVertices());
    // store products of interaction models, i.e. simHits
    for(auto & interactionModel : interactionModels_)
    {
		interactionModel->storeProducts(iEvent);
    }
}

void
FastSimProducer::endStream()
{
	_randomEngine.reset();
}

DEFINE_FWK_MODULE(FastSimProducer);
