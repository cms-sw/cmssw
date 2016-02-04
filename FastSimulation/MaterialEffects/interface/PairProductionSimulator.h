#ifndef PAIRPRODUCTIONSIMULATOR_H
#define FPAIRPRODUCTIONSIMULATOR_H

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsSimulator.h"

/** 
 * This class computes the probability for photons ( under the form 
 * of a ParticlePropagator, i.e., a RawParticle) to convert into an
 * e+e- pair in the tracker layer. In case, it returns a list of 
 * two RawParticle's (e+ and e-). The fraction of radiation lengths 
 * traversed by the particle in this tracker layer is determined in 
 * MaterialEffectsSimulator.
 *
 * This version (a la PDG) of a dE/dx generator replaces the buggy 
 * GEANT3 Fortran -> C++ former version (up to FAMOS_0_8_0_pre7).
 *
 * \author Patrick Janot
 * $Date: 24-Dec-2003
 */ 

class ParticlePropagator;
class RandomEngine;

class PairProductionSimulator : public MaterialEffectsSimulator
{
 public:

  /// Constructor
  PairProductionSimulator(double photonEnergyCut,
			const RandomEngine* engine);

  /// Default Destructor
  ~PairProductionSimulator() {}

 private:

  /// The minimal photon energy for possible conversion
  double photonEnergy;

  /// Generate an e+e- pair according to the probability that it happens
  void compute(ParticlePropagator& Particle);

  /// A universal angular distribution - still from GEANT.
  double gbteth(double ener,double partm,double efrac);
};
#endif
