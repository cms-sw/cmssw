#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

KineParticleFilter::KineParticleFilter(const edm::ParameterSet & cfg)
{
    // Charged particles must have pt greater than chargedPtMin [GeV]
    double chargedPtMin  = cfg.getParameter<double>("chargedPtMin");
    chargedPtMin2  = chargedPtMin*chargedPtMin;

    // Particles must have energy greater than EMin [GeV]
    EMin   = cfg.getParameter<double>("EMin");

    // Allow *ALL* protons with energy > protonEMin
    protonEMin   = cfg.getParameter<double>("protonEMin");

    // Particles must have abs(eta) < etaMax (if close enough to 0,0,0)
    double etaMax = cfg.getParameter<double>("etaMax");
    cos2ThetaMax = (std::exp(2.*etaMax)-1.) / (std::exp(2.*etaMax)+1.);
    cos2ThetaMax *= cos2ThetaMax;

    // Particles must have vertex inside the volume enclosed by ECAL
    double vertexRMax = cfg.getParameter<double>("rMax");
    vertexRMax2 = vertexRMax*vertexRMax;
    vertexZMax = cfg.getParameter<double>("zMax");

}

bool KineParticleFilter::acceptParticle(const RawParticle & particle) const
{

    int pId = abs(particle.pid());

    // skipp invisible particles
    if(pId == 12 || pId == 14 || pId == 16 || pId == 1000022)
    {
	return false;
    }

    // keep all high-energy protons
    else if(pId == 2212 && particle.E() >= protonEMin)
    {
	return true;
    }

    // cut on the energy
    else if( particle.E() < EMin)
    {
	return false;
    }

    // cut on pt of charged particles
    else if( particle.charge()!=0 && particle.Perp2()<chargedPtMin2)
    {
	return false;
    }

    // cut on eta if the origin vertex is close to the beam
    else if( particle.vertex().Perp2() < 25. && particle.cos2Theta() > cos2ThetaMax)
    {
	return false;
    }



    // particles must have vertex in volume enclosed by ECAL
    return acceptVertex(particle.vertex());
}


bool KineParticleFilter::acceptVertex(const XYZTLorentzVector & vertex) const
{
    return vertex.Perp2() < vertexRMax2 && fabs(vertex.Z()) < vertexZMax;
}
