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
#include "FastSimulation/Geometry/interface/Geometry.h"
#include "FastSimulation/Layer/interface/Layer.h"
#include "FastSimulation/Decayer/interface/Decayer.h"
#include "FastSimulation/Propagation/interface/LayerNavigator.h"
#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FastSimulation/FastSimProducer/interface/ParticleFilter.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"  // TODO: get rid of this
#include "FastSimulation/InteractionModel/interface/InteractionModel.h"
#include "FastSimulation/InteractionModel/interface/InteractionModelFactory.h"
#include "FastSimulation/FastSimProducer/interface/ParticleLooper.h"

// other

class FastSimProducer : public edm::stream::EDProducer<> {
public:

    explicit FastSimProducer(const edm::ParameterSet&);
    ~FastSimProducer(){;}

private:

    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    edm::EDGetTokenT<edm::HepMCProduct> genParticlesToken_;
    fastsim::Geometry geometry_;
    double beamPipeRadius_;
    fastsim::ParticleFilter particleFilter_;
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
FastSimProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    LogDebug(MESSAGECATEGORY) << "   produce";

    // do the iov thing
    // TODO: check me
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

    // ?? is this the right place ??
    RandomEngineAndDistribution random(iEvent.streamID());

    fastsim::ParticleLooper particleLooper(
	*genParticles->GetEvent()
	,*pdt
	,beamPipeRadius_
	,particleFilter_
	,output_simTracks
	,output_simVertices);
	
    // loop over particles
    LogDebug(MESSAGECATEGORY) << "################################"
			      << "\n###############################";    

    for(std::unique_ptr<fastsim::Particle> particle = particleLooper.nextParticle(random); particle != 0;particle=particleLooper.nextParticle(random)) 
    {
    	LogDebug(MESSAGECATEGORY) << "\n   moving NEXT particle: " << *particle;

		// move the particle through the layers
		fastsim::LayerNavigator layerNavigator(geometry_);
		const fastsim::Layer * layer = 0;
		while(layerNavigator.moveParticleToNextLayer(*particle,layer))
		{
		    LogDebug(MESSAGECATEGORY) << "   moved to next layer: " << *layer
					      << "\n   new state: " << *particle;

			//std::cout << *particle << std::endl;
			//if(layer) std::cout << layer->getMagneticFieldZ(particle->position()) << std::endl;
		    
		    // perform interaction between layer and particle
		    for(fastsim::InteractionModel * interactionModel : layer->getInteractionModels())
		    {
				LogDebug(MESSAGECATEGORY) << "   interact with " << *interactionModel;
				std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
				interactionModel->interact(*particle,*layer,secondaries,random);
				particleLooper.addSecondaries(particle->position(),particle->simTrackIndex(),secondaries);
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

		// do decays
		if(!particle->isStable() && particle->remainingProperLifeTime() < 1E-20)
		{
		    LogDebug(MESSAGECATEGORY) << "Decaying particle...";
		    std::vector<std::unique_ptr<fastsim::Particle> > secondaries;
		    decayer_.decay(*particle,secondaries,random.theEngine());
		    LogDebug(MESSAGECATEGORY) << "   decay has " << secondaries.size() << " products";
		    particleLooper.addSecondaries(particle->position(),particle->simTrackIndex(),secondaries);
		}
		
		LogDebug(MESSAGECATEGORY) << "################################"
					  << "\n###############################";
    }

    // store simHits and simTracks
    iEvent.put(particleLooper.harvestSimTracks());
    iEvent.put(particleLooper.harvestSimVertices());
    // store products of interaction models, i.e. simHits
    for(auto & interactionModel : interactionModels_)
    {
		interactionModel->storeProducts(iEvent);
    }
}

// TODO: this should actually become a member function of FSimEvent
//       to be used by decays as well
/*
void FastSimProducer::addSecondaries(const fastsim::Particle parent,int parentIndex,std::vector<fastsim::Particle> secondaries)
{
    // add secondaries to the event
    if(secondaries.size() > 0)
    {
	// TODO: probably want to get rid of the TLorentzVector in Particle etc.
	//    see https://root.cern.ch/root/html/MATH_GENVECTOR_Index.html
	//    it got all kinds of methods to do transformations
	int vertexIndex = simEvent_.addSimVertex(math::XYZTLorentzVector(parent.position().X(),
									 parent.position().Y(),
									 parent.position().Z(),
									 parent.position().T())
						 simTrackIndex);
	// TODO: deal with closest charged daughter
	for(const auto & secondary : secondaries)
	{
	    RawParticle _secondary(secondary.pdgId(),
				   math::XYZTLorentzVector(secondary.momentum.X(),
							   secondary.momentum.Y()
							   secondary.momentum.Z()
							   secondary.momentum.E()));
	    simEvent_.addSimTrack(_secondary,vertexIndex);
	}
    }
}
*/
DEFINE_FWK_MODULE(FastSimProducer);
