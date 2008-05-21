#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
//#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"

#include <cmath>

EnergyLossSimulator::EnergyLossSimulator(const RandomEngine* engine,
					 double A, double Z, double density, double radLen) :
    MaterialEffectsSimulator(engine,A,Z,density,radLen) 
{

  theGenerator = new LandauFluctuationGenerator(engine);

}

EnergyLossSimulator::~EnergyLossSimulator() {

  delete theGenerator;

}

void 
EnergyLossSimulator::compute(ParticlePropagator &Particle)
{

  //  FamosHistos* myHistos = FamosHistos::instance();

  double gamma_e = 0.577215664901532861;  // Euler constant

  // The thickness in cm
  double thick = radLengths * radLenIncm();
  
  // This is a simple version (a la PDG) of a dE/dx generator.
  // It replaces the buggy GEANT3 -> C++ former version.
  // Author : Patrick Janot - 8-Jan-2004

  double p2  = Particle.Vect().Mag2();
  double m2  = Particle.mass() * Particle.mass();
  double e2  = p2+m2;

  double beta2 = p2/e2;
  double gama2 = e2/m2;
  
  double charge2 = Particle.charge() * Particle.charge();
  
  // Energy loss spread in GeV
  double eSpread  = 0.1536E-3*charge2*(theZ()/theA())*rho()*thick/beta2;
  
  // Most probable energy loss (from the integrated Bethe-Bloch equation)
  double mostProbableLoss = 
    eSpread * ( log ( 2.*eMass()*beta2*gama2*eSpread/thick
		      / (excitE()*excitE()) )
		- beta2 + 1. - gamma_e );
  
  // Generate the energy loss with Landau fluctuations
  double dedx = mostProbableLoss + eSpread * theGenerator->landau();

  // Compute the new energy and momentum
  double newE = std::max(Particle.mass(),Particle.e()-dedx);
  double fac  = std::sqrt((newE*newE-m2)/p2);
  
  // Update the momentum
  Particle.SetXYZT(Particle.Px()*fac,Particle.Py()*fac, 
		   Particle.Pz()*fac,newE);
  
}

