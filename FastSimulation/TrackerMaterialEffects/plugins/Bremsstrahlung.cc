#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/NewParticle/interface/Particle.h"
#include "FastSimulation/Layer/interface/Layer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <memory>

#include <Math/RotationY.h>
#include <Math/RotationZ.h>

#include "FastSimulation/InteractionModel/interface/InteractionModelFactory.h"
#include "FastSimulation/InteractionModel/interface/InteractionModel.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace fastsim
{
    class Bremsstrahlung : public InteractionModel
    {
    public:
	Bremsstrahlung(const std::string & name,const edm::ParameterSet & cfg);
	void interact(Particle & particle,const Layer & layer,std::vector<std::unique_ptr<Particle> > & secondaries,const RandomEngineAndDistribution & random);
    private:
	math::XYZTLorentzVector brem(Particle & particle , double xmin,const RandomEngineAndDistribution & random) const;
	double gbteth(const double ener,
		      const double partm,
		      const double efrac,
		      const RandomEngineAndDistribution & random) const ;
	// why do we have a dedicated implementation here? check it, probably it can go...
	unsigned int poisson(double ymu, const RandomEngineAndDistribution & random);
	double minPhotonEnergy_;
	double minPhotonEnergyFraction_;
    };
}

fastsim::Bremsstrahlung::Bremsstrahlung(const std::string & name,const edm::ParameterSet & cfg)
    : fastsim::InteractionModel(name)
{
    // Set the minimal photon energy for a Brem from e+/-
    minPhotonEnergy_ = cfg.getParameter<double>("minPhotonEnergy");
    minPhotonEnergyFraction_ = cfg.getParameter<double>("minPhotonEnergyFraction");
}


void fastsim::Bremsstrahlung::interact(fastsim::Particle & particle, const Layer & layer,std::vector<std::unique_ptr<fastsim::Particle> > & secondaries,const RandomEngineAndDistribution & random)
{
    // only consider electrons and positrons
    if(abs(particle.pdgId())!=11)
    {
	return;
    }
    
    double radLengths = layer.getThickness(particle.position(),particle.momentum());

    // Protection : Just stop the electron if more than 1 radiation lengths.
    // This case corresponds to an electron entering the layer parallel to 
    // the layer axis - no reliable simulation can be done in that case...
    if ( radLengths > 4. ) 
    {
	particle.momentum().SetXYZT(0.,0.,0.,0.);
	return;
    }

    // electron must have more energy than minimum photon energy
    if (particle.momentum().E()<minPhotonEnergy_)
    {
	return;
    }

    // Hard brem probability with a photon Energy above threshold.
    double xmin = std::max(minPhotonEnergy_/particle.momentum().E(),minPhotonEnergyFraction_);
    if ( xmin >=1. || xmin <=0. ) 
    {
	return;
    }
    double bremProba = radLengths * ( 4./3. * std::log(1./xmin)
				      - 4./3. * (1.-xmin)
				      + 1./2. * (1.-xmin*xmin) );
    
  
    // Number of photons to be radiated.
    unsigned int nPhotons = poisson(bremProba, random);
    if ( nPhotons == 0) 
    {
	return;
    }

    //Rotate to the lab frame
    double theta = particle.momentum().Theta();
    double phi = particle.momentum().Phi();
    
    // Energy of these photons
    for ( unsigned int i=0; i<nPhotons; ++i ) 
    {
	// Check that there is enough energy left.
	if ( particle.momentum().E() < minPhotonEnergy_ ) break;

	// Add a photon
	secondaries.emplace_back(new fastsim::Particle(22,particle.position(),brem(particle, xmin, random)));
	secondaries.back()->momentum() = ROOT::Math::RotationY(theta)*secondaries.back()->momentum();
	secondaries.back()->momentum() = ROOT::Math::RotationZ(phi)*secondaries.back()->momentum();
	
	// Update the original e+/-
	particle.momentum() -= secondaries.back()->momentum();
    }
}	


math::XYZTLorentzVector
fastsim::Bremsstrahlung::brem(fastsim::Particle & particle , double xmin,const RandomEngineAndDistribution & random) const 
{

    // This is a simple version (a la PDG) of a Brem generator.
    // It replaces the buggy GEANT3 -> C++ former version.
    // Author : Patrick Janot - 25-Dec-2003
    double emass = 0.0005109990615;
    double xp=0;
    double weight = 0.;
  
    do {
	xp = xmin * std::exp ( -std::log(xmin) * random.flatShoot() );
	weight = 1. - xp + 3./4.*xp*xp;
    } while ( weight < random.flatShoot() );
  
  
    // Have photon energy. Now generate angles with respect to the z axis 
    // defined by the incoming particle's momentum.

    // Isotropic in phi
    const double phi = random.flatShoot()*2*M_PI;
    // theta from universal distribution
    const double theta = gbteth(particle.momentum().E(),emass,xp,random)*emass/particle.momentum().E();
  
    // Make momentum components
    double stheta = std::sin(theta);
    double ctheta = std::cos(theta);
    double sphi   = std::sin(phi);
    double cphi   = std::cos(phi);
  
    return xp * particle.momentum().E() * math::XYZTLorentzVector(stheta*cphi,stheta*sphi,ctheta,1.);
  
}

double
fastsim::Bremsstrahlung::gbteth(const double ener,
				const double partm,
				const double efrac,
                                const RandomEngineAndDistribution & random) const 
{
    const double alfa = 0.625;
    
    int Z = 14; // silicon
    const double d = 0.13*(0.8+1.3/Z)*(100.0+(1.0/ener))*(1.0+efrac);
    const double w1 = 9.0/(9.0+d);
    const double umax = ener*M_PI/partm;
    double u;
    
    do 
    {
	double beta = (random.flatShoot()<=w1) ? alfa : 3.0*alfa;
	u = -std::log(random.flatShoot()*random.flatShoot())/beta;
    } 
    while (u>=umax);

    return u;
}


unsigned int 
fastsim::Bremsstrahlung::poisson(double ymu, const RandomEngineAndDistribution & random) 
{
    unsigned int n = 0;
    double prob = std::exp(-ymu);
    double proba = prob;
    double x = random.flatShoot();
    
    while ( proba <= x ) {
	prob *= ymu / double(++n);
	proba += prob;
    }
    
    return n;                                                        
}

DEFINE_EDM_PLUGIN(
    fastsim::InteractionModelFactory,
    fastsim::Bremsstrahlung,
    "fastsim::Bremsstrahlung"
    );
