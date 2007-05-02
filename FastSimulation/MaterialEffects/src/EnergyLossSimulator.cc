#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"

//#include "FamosGeneric/FamosUtils/interface/FamosHistos.h"

#include "CLHEP/Units/PhysicalConstants.h"

#include <iostream>
#include <cmath>
#include <cstring>

EnergyLossSimulator::EnergyLossSimulator(const RandomEngine* engine) :
    MaterialEffectsSimulator(engine) 
{

  theGenerator = new LandauFluctuationGenerator(engine);

}

EnergyLossSimulator::~EnergyLossSimulator() {

  delete theGenerator;

}

void EnergyLossSimulator::compute(ParticlePropagator &Particle)
{

  //  FamosHistos* myHistos = FamosHistos::instance();

  double gamma_e = 0.577215664901532861;  // Euler constant

  // The thickness in cm
  double thick = radLengths * radLenIncm();
  
  // This is a simple version (a la PDG) of a dE/dx generator.
  // It replaces the buggy GEANT3 -> C++ former version.
  // Author : Patrick Janot - 8-Jan-2004

  double p    = Particle.vect().mag();
  double mass = Particle.mass();
  double e    = std::sqrt(p*p+mass*mass);

  double beta2 = p/e;
  double gama2 = e/mass;
  beta2 *= beta2;
  gama2 *= gama2;
  
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

  /*
  myHistos->fill("h100",log10(Particle.vect().mag()),
		        log10(1000.*mostProbableLoss/rho()/thick));
  myHistos->fill("h101",log10(Particle.vect().mag()),
		        log10(1000.*dedx/rho()/thick));
  */

  // Compute the new energy and momentum
  double newEnergy = std::max(mass,e-dedx);
  double fac   = std::sqrt(newEnergy*newEnergy-mass*mass)/p;
  
  // Update the momentum
  Particle.setPx(Particle.px()*fac); 
  Particle.setPy(Particle.py()*fac); 
  Particle.setPz(Particle.pz()*fac);
  Particle.setE(newEnergy);
  
}

