#ifndef ENERGYLOSSUPDATOR_H
#define ENERGYLOSSUPDATOR_H

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsUpdator.h"

/** 
 * This class computes the most probable energy loss by ionization,
 * from a charged particle (under the form of a ParticlePropagator, 
 * i.e., a RawParticle) in the tracker layer, smears it with Landau
 * fluctuations and returns the RawParticle with modified energy. 
 * The tracker material is assumed to be 100% Si - crude approximation - 
 * and the fraction of radiation lengths traversed by the particle 
 * in this tracker layer is determined in MaterialEffectsUpdator.
 *
 * This version (a la PDG) of a dE/dx generator replaces the buggy 
 * GEANT3 Fortran -> C++ former version (up to FAMOS_0_8_0_pre7).
 *
 * \author Patrick Janot
 * $Date: 8-Jan-2004
 */ 


class ParticlePropagator;
class RandomEngine;
class LandauFluctuationGenerator;

class EnergyLossUpdator : public MaterialEffectsUpdator
{
 public:

  /// Constructor
  EnergyLossUpdator(const RandomEngine* engine);

  /// Default Destructor
  ~EnergyLossUpdator();

 private:
  /// The Landau Fluctuation generator
  LandauFluctuationGenerator* theGenerator;

  /// The real dE/dx generation and particle update
  void compute(ParticlePropagator &Particle);
};

#endif
