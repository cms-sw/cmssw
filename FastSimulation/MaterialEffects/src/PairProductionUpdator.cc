#include "FastSimulation/MaterialEffects/interface/PairProductionUpdator.h"

#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <cmath>

using namespace std;

PairProductionUpdator::PairProductionUpdator(double photonEnergyCut,
					     const RandomEngine* engine) :
  MaterialEffectsUpdator(engine) 
{

  // Set the minimal photon energy for possible conversion 
  photonEnergy = std::max(0.100,photonEnergyCut);

}


void PairProductionUpdator::compute(ParticlePropagator& Particle)
{

  double eGamma = Particle.vect().mag(); 

  // The photon has enough energy to create a pair
  if ( eGamma>=photonEnergy ) { 

    // This is a simple version (a la PDG) of a photon conversion generator.
    // It replaces the buggy GEANT3 -> C++ former version.
    // Author : Patrick Janot - 7-Jan-2004

    // Probability to convert is 7/9*(dx/X0)
    if ( -log(random->flatShoot()) <= (7./9.)*radLengths ) {
      
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
      double pElectron = sqrt(max((eElectron+eMass())*tElectron,0.));

      double ePositron = eGamma-eElectron;
      double tPositron = ePositron-eMass();
      double pPositron = sqrt((ePositron+eMass())*tPositron);
      
      // Generate angles
      double phi    = random->flatShoot()*2.*M_PI;
      double sphi   = sin(phi);
      double cphi   = cos(phi);

      double stheta1, stheta2, ctheta1, ctheta2;

      if ( eElectron > ePositron ) {
	double theta1  = gbteth(eElectron,eMass(),xe)*eMass()/eElectron;
	stheta1 = sin(theta1);
	ctheta1 = cos(theta1);
	stheta2 = stheta1*pElectron/pPositron;
	ctheta2 = sqrt(max(0.,1.0-(stheta2*stheta2)));
      } else {
	double theta2  = gbteth(ePositron,eMass(),xe)*eMass()/ePositron;
	stheta2 = sin(theta2);
	ctheta2 = cos(theta2);
	stheta1 = stheta2*pPositron/pElectron;
	ctheta1 = sqrt(max(0.,1.0-(stheta1*stheta1)));
      }
      
      
      double chi = Particle.theta();
      double psi = Particle.phi();
      
      HepLorentzVector PartP(pElectron*stheta1*cphi,
			     pElectron*stheta1*sphi,
			     pElectron*ctheta1,eElectron);
      
      PartP = PartP.rotateY(chi);
      PartP = PartP.rotateZ(psi);
      
      RawParticle * E1 = new RawParticle(11,PartP);
      
      // Create the positron
      HepLorentzVector PartP2(-pPositron*stheta2*cphi,
			      -pPositron*stheta2*sphi,
	  		       pPositron*ctheta2,ePositron);
      
      PartP2 = PartP2.rotateY(chi);
      PartP2 = PartP2.rotateZ(psi);
      
      RawParticle * E2 =new  RawParticle (-11,PartP2);
      
      _theUpdatedState.push_back(E1);
      _theUpdatedState.push_back(E2);

    }
  }
}

double PairProductionUpdator::gbteth(double ener,double partm,double efrac)
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
      u = -(log(random->flatShoot()*random->flatShoot()))/beta;
  } while (u>=umax);

  return u;
}
