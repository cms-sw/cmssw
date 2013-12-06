#include "FastSimulation/MaterialEffects/interface/PairProductionSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"

#include <cmath>

PairProductionSimulator::PairProductionSimulator(double photonEnergyCut)
{
  // Set the minimal photon energy for possible conversion 
  photonEnergy = std::max(0.100,photonEnergyCut);  
}

void 
PairProductionSimulator::compute(ParticlePropagator& Particle, RandomEngineAndDistribution const* random)
{

  double eGamma = Particle.e(); 

  // The photon has enough energy to create a pair
  if ( eGamma>=photonEnergy ) { 

    // This is a simple version (a la PDG) of a photon conversion generator.
    // It replaces the buggy GEANT3 -> C++ former version.
    // Author : Patrick Janot - 7-Jan-2004

    // Probability to convert is 7/9*(dx/X0)
    if ( -std::log(random->flatShoot()) <= (7./9.)*radLengths ) {
      
      double xe=0;
      double xm=eMass()/eGamma;
      double weight = 0.;
  
      // Generate electron energy between emass and eGamma-emass
      do {
	xe = random->flatShoot()*(1.-2.*xm) + xm;
	weight = 1. - 4./3.*xe*(1.-xe);
      } while ( weight < random->flatShoot() );
  
      double eElectron = xe * eGamma;
      double tElectron = eElectron-eMass();
      double pElectron = std::sqrt(std::max((eElectron+eMass())*tElectron,0.));

      double ePositron = eGamma-eElectron;
      double tPositron = ePositron-eMass();
      double pPositron = std::sqrt((ePositron+eMass())*tPositron);
      
      // Generate angles
      double phi    = random->flatShoot()*2.*M_PI;
      double sphi   = std::sin(phi);
      double cphi   = std::cos(phi);

      double stheta1, stheta2, ctheta1, ctheta2;

      if ( eElectron > ePositron ) {
	double theta1  = gbteth(eElectron,eMass(),xe,random)*eMass()/eElectron;
	stheta1 = std::sin(theta1);
	ctheta1 = std::cos(theta1);
	stheta2 = stheta1*pElectron/pPositron;
	ctheta2 = std::sqrt(std::max(0.,1.0-(stheta2*stheta2)));
      } else {
	double theta2  = gbteth(ePositron,eMass(),xe,random)*eMass()/ePositron;
	stheta2 = std::sin(theta2);
	ctheta2 = std::cos(theta2);
	stheta1 = stheta2*pPositron/pElectron;
	ctheta1 = std::sqrt(std::max(0.,1.0-(stheta1*stheta1)));
      }
      
      
      double chi = Particle.theta();
      double psi = Particle.phi();
      RawParticle::RotationZ rotZ(psi);
      RawParticle::RotationY rotY(chi);
     
      _theUpdatedState.resize(2,RawParticle());

      // The eletron
      _theUpdatedState[0].SetXYZT(pElectron*stheta1*cphi,
				  pElectron*stheta1*sphi,
				  pElectron*ctheta1,
				  eElectron);
      _theUpdatedState[0].setID(+11);
      _theUpdatedState[0].rotate(rotY);
      _theUpdatedState[0].rotate(rotZ);
      
      // The positron
      _theUpdatedState[1].SetXYZT(-pPositron*stheta2*cphi,
				  -pPositron*stheta2*sphi,
				   pPositron*ctheta2,
				   ePositron);
      _theUpdatedState[1].setID(-11);
      _theUpdatedState[1].rotate(rotY);
      _theUpdatedState[1].rotate(rotZ);
      
    }
  } 
}

double 
PairProductionSimulator::gbteth(double ener,double partm,double efrac, RandomEngineAndDistribution const* random)
{
  const double alfa = 0.625;

  double d = 0.13*(0.8+1.3/theZ())*(100.0+(1.0/ener))*(1.0+efrac);
  double w1 = 9.0/(9.0+d);
  double umax = ener*M_PI/partm;
  double u;

  do {
      double beta;
      if (random->flatShoot()<=w1) beta = alfa;
      else beta = 3.0*alfa;
      u = -(std::log(random->flatShoot()*random->flatShoot()))/beta;
  } while (u>=umax);

  return u;
}
