#ifndef MULTIPLESCATTERINGSIMULATOR_H
#define MULTIPLESCATTERINGSIMULATOR_H

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsSimulator.h"

/** 
 * This class computes the direction change by multiple scattering 
 * of a charged particle (under the form of a ParticlePropagator, 
 * i.e., a RawParticle) in the tracker layer, and returns the 
 * RawParticle with the modified momentum direction. The tracker 
 * material is assumed to be 100% Si and the Tracker layers are 
 * assumed infinitely thin. The fraction of radiation lengths 
 * traversed by the particle in this tracker layer is determined 
 * in MaterialEffectsSimulator.
 *
 * This version (a la PDG) of a multiple scattering simulator replaces 
 * the buggy GEANT3 Fortran -> C++ former version (up to FAMOS_0_8_0_pre7).
 *
 * \author Patrick Janot
 * $Date: 8-Jan-2004
 */ 

class ParticlePropagator;
class RandomEngine;

class MultipleScatteringSimulator : public MaterialEffectsSimulator
{
 public:

  /// Default Constructor
  MultipleScatteringSimulator(const RandomEngine* engine);

  /// Default Destructor
  ~MultipleScatteringSimulator() {} ;

 private:

  /// The real dE/dx generation and particle update
  void compute(ParticlePropagator &Particle);

 private: 
  
  /// Save (a tiny bit of) time
  double sqr12;

};

#endif
