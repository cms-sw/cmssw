#ifndef BREMSSTRAHLUNGUPDATOR_H
#define BREMSSTRAHLUNGUPDATOR_H

#include "CLHEP/Vector/LorentzVector.h"

#include "FastSimulation/MaterialEffects/interface/MaterialEffectsUpdator.h"

/** 
 * This class computes the number, energy and angles of Bremsstrahlung 
 * photons emitted by electrons and positrons (under the form of a 
 * ParticlePropagator, i.e., a RawParticle) in the tracker layer, 
 * and returns the RawParticle modified after radiation as well as 
 * a list of photons (i.e., a list of RawParticles as well).
 * The fraction of radiation lengths traversed by the particle 
 * in this tracker layer is determined in MaterialEffectsUpdator.
 *
 * This version (a la PDG) of a dE/dx generator replaces the buggy 
 * GEANT3 Fortran -> C++ former version (up to FAMOS_0_8_0_pre7).
 *
 * \author Patrick Janot
 * $Date: 25-Dec-2003
 */ 


class ParticlePropagator;
class RandomEngine;

class BremsstrahlungUpdator : public MaterialEffectsUpdator
{
 public:

  /// Constructor
  BremsstrahlungUpdator(double photonEnergyCut, 
			double photonFractECut,
			const RandomEngine* engine); 

  /// Default destructor
  ~BremsstrahlungUpdator() {}

 private:
  
  /// The minimum photon energy to be radiated, in GeV
  double photonEnergy;

  /// The minimum photon fractional energy (wrt that of the electron)
  double photonFractE;

  /// The fractional photon energy cut (determined from the above two)
  double xmin;

  /// Generate numbers according to a Poisson distribution of mean ymu.
  unsigned int poisson(double ymu);

  /// Generate Bremsstrahlung photons
  void compute(ParticlePropagator &Particle);

  /// Compute Brem photon energy and angles, if any.
  HepLorentzVector brem(HepLorentzVector p);

  /// A universal angular distribution - still from GEANT.
  double gbteth(const double ener,const double partm,const double efrac) const;

};
#endif
