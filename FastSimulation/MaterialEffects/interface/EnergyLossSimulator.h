#ifndef ENERGYLOSSSIMULATOR_H
#define ENERGYLOSSSIMULATOR_H

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsSimulator.h"

/** 
 * This class computes the most probable energy loss by ionization,
 * from a charged particle (under the form of a ParticlePropagator, 
 * i.e., a RawParticle) in the tracker layer, smears it with Landau
 * fluctuations and returns the RawParticle with modified energy. 
 * The tracker material is assumed to be 100% Si - crude approximation - 
 * and the fraction of radiation lengths traversed by the particle 
 * in this tracker layer is determined in MaterialEffectsSimulator.
 *
 * This version (a la PDG) of a dE/dx generator replaces the buggy 
 * GEANT3 Fortran -> C++ former version (up to FAMOS_0_8_0_pre7).
 *
 * \author Patrick Janot
 * $Date: 8-Jan-2004
 */ 


class RandomEngineAndDistribution;
class LandauFluctuationGenerator;

class EnergyLossSimulator : public MaterialEffectsSimulator
{
 public:

  /// Constructor
  EnergyLossSimulator(double A, double Z, double density, double radLen);

  /// Default Destructor
  ~EnergyLossSimulator() override;

  /// Return most probable energy loss
  inline double mostLikelyLoss() const { return mostProbableLoss; }

  /// Returns the actual energy lost
  inline const XYZTLorentzVector& deltaMom() const { return deltaP; }

 private:
  /// The Landau Fluctuation generator
  LandauFluctuationGenerator* theGenerator;

  /// The real dE/dx generation and particle update
  void compute(ParticlePropagator &Particle, RandomEngineAndDistribution const*) override;

  /// The most probable enery loss
  double mostProbableLoss;

  /// The actual energy loss
  XYZTLorentzVector deltaP;

};

#endif
