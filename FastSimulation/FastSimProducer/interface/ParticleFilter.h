#ifndef FASTSIM_PARTICLEFILTER
#define FASTSIM_PARTICLEFILTER

#include "DataFormats/Math/interface/LorentzVector.h"

namespace edm
{
    class ParameterSet;
}

namespace fastsim
{
    class Particle;
    class ParticleFilter
    {
    public:
	ParticleFilter(const edm::ParameterSet & cfg);
	bool accepts(const Particle & particle) const;
	bool accepts(const math::XYZTLorentzVector & originVertexPosition) const;

    private:
	// see constructor for comments
	double chargedPtMin2_, EMin_, protonEMin_;
	double cos2ThetaMax_;
	double vertexRMax2_,vertexZMax_;
    };
}

#endif
