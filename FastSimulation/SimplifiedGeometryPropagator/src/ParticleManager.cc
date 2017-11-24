#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleManager.h"

#include "HepMC/GenEvent.h"
#include "HepMC/Units.h"
#include "HepPDT/ParticleDataTable.hh"

#include "FastSimulation/SimplifiedGeometryPropagator/interface/Particle.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/ParticleFilter.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Constants.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/SimplifiedGeometry.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

fastsim::ParticleManager::ParticleManager(
    const HepMC::GenEvent & genEvent,
    const HepPDT::ParticleDataTable & particleDataTable,
    double beamPipeRadius,
    double deltaRchargedMother,
    const fastsim::ParticleFilter & particleFilter,
    std::vector<SimTrack> & simTracks,
    std::vector<SimVertex> & simVertices)
    : genEvent_(&genEvent)
    , genParticleIterator_(genEvent_->particles_begin())
    , genParticleEnd_(genEvent_->particles_end())
    , genParticleIndex_(1)
    , particleDataTable_(&particleDataTable)
    , beamPipeRadius2_(beamPipeRadius*beamPipeRadius)
    , deltaRchargedMother_(deltaRchargedMother)
    , particleFilter_(&particleFilter)
    , simTracks_(&simTracks)
    , simVertices_(&simVertices)
    // prepare unit convsersions
    //  --------------------------------------------
    // |          |      hepmc               |  cms |
    //  --------------------------------------------
    // | length   | genEvent_->length_unit   |  cm  |
    // | momentum | genEvent_->momentum_unit |  GeV |
    // | time     | length unit (t*c)        |  ns  |
    //  --------------------------------------------
    , momentumUnitConversionFactor_(conversion_factor( genEvent_->momentum_unit(), HepMC::Units::GEV ))
    , lengthUnitConversionFactor_(conversion_factor(genEvent_->length_unit(),HepMC::Units::LengthUnit::CM))
    , lengthUnitConversionFactor2_(lengthUnitConversionFactor_*lengthUnitConversionFactor_)
    , timeUnitConversionFactor_(lengthUnitConversionFactor_/fastsim::Constants::speedOfLight)

{

    // add the main vertex from the signal event to the simvertex collection
    if(genEvent.vertices_begin() != genEvent_->vertices_end())
    {
    	const HepMC::FourVector & position = (*genEvent.vertices_begin())->position();
    	addSimVertex(math::XYZTLorentzVector(position.x()*lengthUnitConversionFactor_,
    					     position.y()*lengthUnitConversionFactor_,
    					     position.z()*lengthUnitConversionFactor_,
    					     position.t()*timeUnitConversionFactor_)
                    ,-1);
    }
}

fastsim::ParticleManager::~ParticleManager(){}

std::unique_ptr<fastsim::Particle> fastsim::ParticleManager::nextParticle(const RandomEngineAndDistribution & random)
{
    std::unique_ptr<fastsim::Particle> particle;

    // retrieve particle from buffer
    if(!particleBuffer_.empty())
    {
    	particle = std::move(particleBuffer_.back());
    	particleBuffer_.pop_back();
    }
    // or from genParticle list
    else
    {
	   particle = nextGenParticle();
       if(!particle) return nullptr;
    }

    // if filter does not accept, skip particle
    if(!particleFilter_->accepts(*particle))
    {
	    return nextParticle(random);
    }

    // lifetime or charge of particle are not yet set
    if(!particle->remainingProperLifeTimeIsSet() || !particle->chargeIsSet())
    {
    	// retrieve the particle data
    	const HepPDT::ParticleData * particleData = particleDataTable_->particle(HepPDT::ParticleID(particle->pdgId()));
    	if(!particleData)
    	{
    	    throw cms::Exception("fastsim::ParticleManager") << "unknown pdg id: " << particle->pdgId() << std::endl;
    	}

    	// set lifetime
    	if(!particle->remainingProperLifeTimeIsSet())
    	{
            // The lifetime is 0. in the Pythia Particle Data Table! Calculate from width instead (ct=hbar/width).
            // ct=particleData->lifetime().value();
            double width = particleData->totalWidth().value();
    	    if(width > 1.0e-35)
    	    {
                particle->setRemainingProperLifeTimeC(-log(random.flatShoot())*6.582119e-25/width/10.); // ct in cm
    	    }
            else
    	    {
                particle->setStable();
    	    }
    	}

    	// set charge
    	if(!particle->chargeIsSet())
    	{
    	    particle->setCharge(particleData->charge());
    	}
    }

    // add corresponding simTrack to simTrack collection
    unsigned simTrackIndex = addSimTrack(particle.get());
    particle->setSimTrackIndex(simTrackIndex);

    // and return
    return particle;
}


void fastsim::ParticleManager::addSecondaries(
    const math::XYZTLorentzVector & vertexPosition,
    int parentSimTrackIndex,
    std::vector<std::unique_ptr<Particle> > & secondaries,
    const SimplifiedGeometry * layer)
{

    // vertex must be within the accepted volume
    if(!particleFilter_->acceptsVtx(vertexPosition))
    {
	   return;
    }

    // no need to create vertex in case no particles are produced
    if(secondaries.empty()){
    	return;
    }

    // add simVertex
    unsigned simVertexIndex = addSimVertex(vertexPosition,parentSimTrackIndex);

    // closest charged daughter continues the track of the mother particle
    // simplified tracking algorithm for fastSim
    double distMin = 99999.;
    int idx = -1;
    int idxMin = -1;
    for(auto & secondary : secondaries)
    {
    	idx++;
        if(secondary->getMotherDeltaR() != -1){
            if(secondary->getMotherDeltaR() > deltaRchargedMother_){
                // larger than max requirement on deltaR
                secondary->resetMother();
            }else{
            	if(secondary->getMotherDeltaR() < distMin){
            		distMin = secondary->getMotherDeltaR();
            		idxMin = idx;
            	}
            }            
        }
    }
    
    // add secondaries to buffer
    idx = -1;
    for(auto & secondary : secondaries)
    {
    	idx++;
    	if(idxMin != -1){
            // reset all but the particle with the lowest deltaR (which is at idxMin)
    		if(secondary->getMotherDeltaR() != -1 && idx != idxMin){
    			secondary->resetMother();
    		}
    	}

        // set origin vertex
    	secondary->setSimVertexIndex(simVertexIndex);
        //
        if(layer)
        {
            secondary->setOnLayer(layer->isForward(), layer->index());
        }
        // ...and add particle to buffer
    	particleBuffer_.push_back(std::move(secondary));
    }

}

unsigned fastsim::ParticleManager::addEndVertex(const fastsim::Particle * particle)
{
    return this->addSimVertex(particle->position(), particle->simTrackIndex());
}

unsigned fastsim::ParticleManager::addSimVertex(
    const math::XYZTLorentzVector & position,
    int parentSimTrackIndex)
{
    int simVertexIndex = simVertices_->size();
    simVertices_->emplace_back(position.Vect(),
			       position.T(),
			       parentSimTrackIndex,
			       simVertexIndex);
    return simVertexIndex;
}

unsigned fastsim::ParticleManager::addSimTrack(const fastsim::Particle * particle)
{
	int simTrackIndex;
	// Again: FastSim cheat tracking -> continue track of mother
	if(particle->getMotherDeltaR() != -1){
		simTrackIndex = particle->getMotherSimTrackIndex();
	}
    // or create new SimTrack
	else
    {
		simTrackIndex = simTracks_->size();
    	simTracks_->emplace_back(particle->pdgId(),
                    particle->momentum(),
                    particle->simVertexIndex(),
                    particle->genParticleIndex());
    	simTracks_->back().setTrackId(simTrackIndex);
	}
    return simTrackIndex;
}

std::unique_ptr<fastsim::Particle> fastsim::ParticleManager::nextGenParticle()
{
    // only consider particles that start in the beam pipe and end outside the beam pipe
    // try to get the decay time from pythia
    // use hepmc units
    // make the link simtrack to simvertex
    // try not to change the simvertex structure
    
    // loop over gen particles
    for ( ; genParticleIterator_ != genParticleEnd_ ; ++genParticleIterator_,++genParticleIndex_ ) 
    {
    	// some handy pointers and references
    	const HepMC::GenParticle & particle = **genParticleIterator_;
    	const HepMC::GenVertex * productionVertex = particle.production_vertex();
    	const HepMC::GenVertex * endVertex = particle.end_vertex();

        // skip incoming particles
        if(!productionVertex){
            continue;
        }

    	// particle must be produced within the beampipe
    	if(productionVertex->position().perp2()*lengthUnitConversionFactor2_ > beamPipeRadius2_)
    	{
    	    continue;
    	}
    	
    	// particle must not decay before it reaches the beam pipe
    	if(endVertex && endVertex->position().perp2()*lengthUnitConversionFactor2_ < beamPipeRadius2_)
    	{
    	    continue;
    	}

    	// make the particle
    	std::unique_ptr<Particle> newParticle(
    	    new Particle(particle.pdg_id(),
    			 math::XYZTLorentzVector(productionVertex->position().x()*lengthUnitConversionFactor_,
    						 productionVertex->position().y()*lengthUnitConversionFactor_,
    						 productionVertex->position().z()*lengthUnitConversionFactor_,
    						 productionVertex->position().t()*timeUnitConversionFactor_),
    			 math::XYZTLorentzVector(particle.momentum().x()*momentumUnitConversionFactor_,
    						 particle.momentum().y()*momentumUnitConversionFactor_,
    						 particle.momentum().z()*momentumUnitConversionFactor_,
    						 particle.momentum().e()*momentumUnitConversionFactor_)));
    	newParticle->setGenParticleIndex(genParticleIndex_);

    	// try to get the life time of the particle from the genEvent
    	if(endVertex)
    	{
    	    double labFrameLifeTime = (endVertex->position().t() - productionVertex->position().t())*timeUnitConversionFactor_;
    	    newParticle->setRemainingProperLifeTimeC(labFrameLifeTime / newParticle->gamma() * fastsim::Constants::speedOfLight);
    	}

        // TODO: The products of a b-decay should point to that vertex and not to the primary vertex!
        // Seems like this information has to be taken from the genEvent. How to do this? Is this really neccessary?
        // See FBaseSimEvent::fill(..)
        newParticle->setSimVertexIndex(0);

        // iterator/index has to be increased in case of return (is not done by the loop then)
        ++genParticleIterator_; ++genParticleIndex_;
    	// and return
    	return std::move(newParticle);
    }

    return std::unique_ptr<Particle>();
}
