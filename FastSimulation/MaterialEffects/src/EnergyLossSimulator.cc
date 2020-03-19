#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
//#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/Utilities/interface/LandauFluctuationGenerator.h"

#include <cmath>

EnergyLossSimulator::EnergyLossSimulator(double A, double Z, double density, double radLen)
    : MaterialEffectsSimulator(A, Z, density, radLen) {
  theGenerator = new LandauFluctuationGenerator();
}

EnergyLossSimulator::~EnergyLossSimulator() { delete theGenerator; }

void EnergyLossSimulator::compute(ParticlePropagator& Particle, RandomEngineAndDistribution const* random) {
  //  FamosHistos* myHistos = FamosHistos::instance();

  // double gamma_e = 0.577215664901532861;  // Euler constant

  // The thickness in cm
  double thick = radLengths * radLenIncm();

  // This is a simple version (a la PDG) of a dE/dx generator.
  // It replaces the buggy GEANT3 -> C++ former version.
  // Author : Patrick Janot - 8-Jan-2004

  double p2 = Particle.particle().Vect().Mag2();
  double verySmallP2 = 0.0001;
  if (p2 <= verySmallP2) {
    deltaP.SetXYZT(0., 0., 0., 0.);
    return;
  }
  double m2 = Particle.particle().mass() * Particle.particle().mass();
  double e2 = p2 + m2;

  double beta2 = p2 / e2;
  double gama2 = e2 / m2;

  double charge2 = Particle.particle().charge() * Particle.particle().charge();

  // Energy loss spread in GeV
  double eSpread = 0.1536E-3 * charge2 * (theZ() / theA()) * rho() * thick / beta2;

  // Most probable energy loss (from the integrated Bethe-Bloch equation)
  mostProbableLoss = eSpread * (log(2. * eMass() * beta2 * gama2 * eSpread / (excitE() * excitE())) - beta2 + 0.200);

  // This one can be needed on output (but is not used internally)
  // meanEnergyLoss = 2.*eSpread * ( log ( 2.*eMass()*beta2*gama2 /excitE() ) - beta2 );

  // Generate the energy loss with Landau fluctuations
  double dedx = mostProbableLoss + eSpread * theGenerator->landau(random);

  // Compute the new energy and momentum
  double aBitAboveMass = Particle.particle().mass() * 1.0001;
  double newE = std::max(aBitAboveMass, Particle.particle().e() - dedx);
  //  double newE = std::max(Particle.particle().mass(),Particle.particle().e()-dedx);
  double fac = std::sqrt((newE * newE - m2) / p2);

  // Update the momentum
  deltaP.SetXYZT(Particle.particle().Px() * (1. - fac),
                 Particle.particle().Py() * (1. - fac),
                 Particle.particle().Pz() * (1. - fac),
                 Particle.particle().E() - newE);
  Particle.particle().setMomentum(
      Particle.particle().Px() * fac, Particle.particle().Py() * fac, Particle.particle().Pz() * fac, newE);
}
