#include "FastSimulation/FastSimProducer/interface/ParticleFilter.h"
#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

fastsim::ParticleFilter::ParticleFilter(const edm::ParameterSet & cfg)
{
    // Charged particles must have pt greater than chargedPtMin [GeV]
    double chargedPtMin  = cfg.getParameter<double>("chargedPtMin");
    chargedPtMin2_  = chargedPtMin*chargedPtMin;
    
    // Particles must have energy greater than EMin [GeV]
    EMin_   = cfg.getParameter<double>("EMin");
    
    // Allow *ALL* protons with energy > protonEMin
    protonEMin_   = cfg.getParameter<double>("protonEMin");
    
    // Particles must have abs(eta) < etaMax (if close enough to 0,0,0)
    double etaMax = cfg.getParameter<double>("etaMax");
    cos2ThetaMax_ = (std::exp(2.*etaMax)-1.) / (std::exp(2.*etaMax)+1.);
    cos2ThetaMax_ *= cos2ThetaMax_;

    // Particles must have vertex inside the volume enclosed by ECAL
    vertexRMax2_ = 129.0*129.0; 
    vertexZMax_ = 317;
}

bool fastsim::ParticleFilter::accepts(const fastsim::Particle & particle) const
{
    int pId = abs(particle.pdgId());

    // skip invisible particles
    if(pId == 12 || pId == 14 || pId == 16 || pId == 1000022)
    {
	return false;
    }
    
    // keep all high-energy protons
    else if(pId == 2212 && particle.momentum().E() >= protonEMin_)
    {
	return true;
    }
    
    // cut on the energy
    else if( particle.momentum().E() < EMin_)
    {
	return false;
    }
    
    // cut on pt of charged particles
    else if( particle.charge()!=0 && particle.momentum().Perp2()<chargedPtMin2_)
    {
	return false;
    }
    
    // cut on eta if the origin vertex is close to the beam
    else if( particle.position().Perp2() < 25. && particle.momentum().Pz()*particle.momentum().Pz()/particle.momentum().P2() > cos2ThetaMax_)
    {
	return false;
    }

    // particles must have vertex in volume enclosed by ECAL
    return accepts(particle.position());
} 


bool fastsim::ParticleFilter::accepts(const math::XYZTLorentzVector & originVertex) const
{
    return originVertex.Perp2() < vertexRMax2_ && fabs(originVertex.Z()) < vertexZMax_;
}
