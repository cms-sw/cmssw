#ifndef FASTSIM_PARTICLELOOPER_H
#define FASTSIM_PARTICLELOOPER_H

#include "DataFormats/Math/interface/LorentzVector.h"
#include "HepMC/GenEvent.h"
#include "vector"
#include "memory"
// TODO: TREAT PARTICLE FILTER PROPERLY

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

namespace HepPDT
{
    class ParticleDataTable;
}

//class SimTrack;
//class SimVertex;
class RandomEngineAndDistribution;

namespace fastsim {
    class Particle;
    class ParticleFilter;
    class ParticleLooper
    {

    public:

	ParticleLooper(
	    const HepMC::GenEvent & genEvent,
	    const HepPDT::ParticleDataTable & particleDataTable,
	    double beamPipeRadius,
	    const ParticleFilter & particleFilter,
	    std::unique_ptr<std::vector<SimTrack> > & simTracks,
	    std::unique_ptr<std::vector<SimVertex> > & simVertices);
	
	~ParticleLooper();

	std::unique_ptr<Particle> nextParticle(const RandomEngineAndDistribution & random);
	
	void addSecondaries(
	    const math::XYZTLorentzVector & vertexPosition,
	    int motherSimTrackId,
	    std::vector<std::unique_ptr<Particle> > & secondaries);

	std::unique_ptr<std::vector<SimTrack> > harvestSimTracks()
	{
	    return std::move(simTracks_);
	}

	std::unique_ptr<std::vector<SimVertex> > harvestSimVertices()
	{
	    return std::move(simVertices_);
	}


    private:

	unsigned addSimVertex(
	    const math::XYZTLorentzVector & position,
	    int motherIndex);
	
	unsigned addSimTrack(const Particle * particle);

	std::unique_ptr<Particle> nextGenParticle();

	// data members
	const HepMC::GenEvent * const genEvent_;
	HepMC::GenEvent::particle_const_iterator genParticleIterator_;
	const HepMC::GenEvent::particle_const_iterator genParticleEnd_;
	int genParticleIndex_;
	const HepPDT::ParticleDataTable * const particleDataTable_;
	const double beamPipeRadius2_;
	const ParticleFilter * const particleFilter_;
	std::unique_ptr<std::vector<SimTrack> > simTracks_;
	std::unique_ptr<std::vector<SimVertex> > simVertices_;
	double momentumUnitConversionFactor_;
	double lengthUnitConversionFactor_;
	double lengthUnitConversionFactor2_;
	double timeUnitConversionFactor_;
	std::vector<std::unique_ptr<Particle> > particleBuffer_;
    };
}

#endif
