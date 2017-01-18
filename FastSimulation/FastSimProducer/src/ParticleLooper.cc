#include "FastSimulation/FastSimProducer/interface/ParticleLooper.h"

#include "HepMC/GenEvent.h"
#include "HepMC/Units.h"
#include "HepPDT/ParticleDataTable.hh"

#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FastSimulation/FastSimProducer/interface/ParticleFilter.h"
#include "FastSimulation/Constants/interface/Constants.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

fastsim::ParticleLooper::ParticleLooper(
    const HepMC::GenEvent & genEvent,
    const HepPDT::ParticleDataTable & particleDataTable,
    double beamPipeRadius,
    const fastsim::ParticleFilter & particleFilter,
    std::unique_ptr<std::vector<SimTrack> > & simTracks,
    std::unique_ptr<std::vector<SimVertex> > & simVertices)
    : genEvent_(&genEvent)
    , genParticleIterator_(genEvent_->particles_begin())
    , genParticleEnd_(genEvent_->particles_end())
    , genParticleIndex_(1)
    , particleDataTable_(&particleDataTable)
    , beamPipeRadius2_(beamPipeRadius*beamPipeRadius)
    , particleFilter_(&particleFilter)
    , simTracks_(std::move(simTracks))
    , simVertices_(std::move(simVertices))
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
    , timeUnitConversionFactor_(lengthUnitConversionFactor_/29.9792458) // speed of light [cm / ns]
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

fastsim::ParticleLooper::~ParticleLooper(){}

std::unique_ptr<fastsim::Particle> fastsim::ParticleLooper::nextParticle(const RandomEngineAndDistribution & random)
{
    std::unique_ptr<fastsim::Particle> particle;

    // retrieve particle from buffer
    if(particleBuffer_.size() > 0)
    {
    	particle = std::move(particleBuffer_.back());
    	particleBuffer_.pop_back();
    }
    // or from genParticle list
    else
    {
	   particle = nextGenParticle();
       if(!particle) return 0;
    }

    // if filter does not accept, skip particle
    if(!particleFilter_->accepts(*particle))
    {
	   return nextParticle(random);
    }
    if(!particle->remainingProperLifeTimeIsSet() || !particle->chargeIsSet() )
    {
    	// retrieve the particle data
    	const HepPDT::ParticleData * particleData = particleDataTable_->particle(HepPDT::ParticleID(particle->pdgId()));
    	if(!particleData)
    	{
    	    throw cms::Exception("fastsim::ParticleLooper") << "unknown pdg id" << std::endl;
    	}

    	// set lifetime
    	if(!particle->remainingProperLifeTimeIsSet())
    	{
    	    double averageLifeTime = particleData->lifetime().value()/fastsim::Constants::speedOfLight; //!!! units: particleData returns units in c*t?
    	    if(averageLifeTime > 1e25 || averageLifeTime < 1e-25) // ridiculously safe // particleData seems to return 0 in case particle is stable!
    	    {
    		  particle->setStable();
    	    }
    	    else
    	    {
    		  particle->setRemainingProperLifeTime(-log(random.flatShoot())*averageLifeTime);
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

    //std::cout << "particle position: " << particle->position().X() << ", " << particle->position().Y() << " [R="<< sqrt(particle->position().X()*particle->position().X()+particle->position().Y()*particle->position().Y()) << "], " << particle->position().Z() << ", " << (particle->position().T())
    //      << "; momentum: " << particle->momentum().Px() << ", " << particle->momentum().Py() << ", " << particle->momentum().Pz() << ", " << particle->momentum().e()
    //      << std::endl;

    // and return
    return particle;
}

// TODO: closest charged daughter...
// NOTE:  decayer and interactions must provide particles with right units
void fastsim::ParticleLooper::addSecondaries(
    const math::XYZTLorentzVector & vertexPosition,
    int parentSimTrackIndex,
    std::vector<std::unique_ptr<Particle> > & secondaries)
{

    // vertex must be within the accepted volume
    if(!particleFilter_->accepts(vertexPosition))
    {
	return;
    }

    // no need to create vertex in case no particles are produced
    if(secondaries.size()==0){
    	return;
    }

    // add simVertex
    unsigned simVertexIndex = addSimVertex(vertexPosition,parentSimTrackIndex);

    // add secondaries to buffer
    for(auto & secondary : secondaries)
    {
	secondary->setSimVertexIndex(simVertexIndex);
	particleBuffer_.push_back(std::move(secondary));
    }

}

unsigned fastsim::ParticleLooper::addSimVertex(
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

unsigned fastsim::ParticleLooper::addSimTrack(const fastsim::Particle * particle)
{
    int simTrackIndex = simTracks_->size();
    simTracks_->emplace_back(particle->pdgId(),
                    particle->momentum(),
                    particle->simVertexIndex(),
                    particle->genParticleIndex());
    simTracks_->back().setTrackId(simTrackIndex);
    return simTrackIndex;
}

std::unique_ptr<fastsim::Particle> fastsim::ParticleLooper::nextGenParticle()
{
    // only consider particles that start in the beam pipe and end outside the beam pipe
    // try to get the decay time from pythia
    // use hepmc units
    // make the link simtrack to simvertex
    // try not to change the simvertex structure
    // print a couple of ttbar events to undertand the simtrack structure? nah...
    
    for ( ; genParticleIterator_ != genParticleEnd_ ; ++genParticleIterator_,++genParticleIndex_ ) // loop over gen particles
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
    	    newParticle->setRemainingProperLifeTime(labFrameLifeTime * newParticle->gamma());
    	}

        // TODO: THIS HAS TO BE FIXED! E.g. the products of a b-decay should point to that vertex and not to the primary vertex!
        newParticle->setSimVertexIndex(0);

        ++genParticleIterator_; ++genParticleIndex_;
    	// and return
    	return std::move(newParticle);
    }

    return std::unique_ptr<Particle>();
}
