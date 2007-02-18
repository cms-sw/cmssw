#ifndef MATERIALEFFECTSUPDATOR_H
#define MATERIALEFFECTSUPDATOR_H

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include <list>
#include <utility>

/** 
 * This is the generic class for Material Effects in the tracker material, 
 * from which FamosPairProductionUpdator, FamosBremsstrahlungUpdator, 
 * FamosEnergyLossUpdator and FamosMultipleScatteringUpdator inherit. It
 * determines the fraction of radiation lengths traversed by the particle 
 * in this tracker layer, defines the tracker layer characteristics (hmmm,
 * 100% Silicon, very crude, but what can we do?) and returns a list of
 * new RawParticles if needed.
 *
 * \author Stephan Wynhoff, Florian Beaudette, Patrick Janot
 * $Date: Last update 8-Jan-2004
 */ 

class MaterialEffectsUpdator
{
 public:

  typedef std::list<const RawParticle*>::const_iterator RHEP_const_iter;

  MaterialEffectsUpdator(const RandomEngine* engine) { 
    random = engine;
    _theUpdatedState.clear(); 
}

  virtual ~MaterialEffectsUpdator() {
    // Don't delete the objects contained in the list
    _theUpdatedState.clear();
  }

  /// Functions to return atomic properties of the material
  /// Here the tracker material is assumed to be 100% Silicon

  /// A
  inline double theA() const { return 28.0855; }
  /// Z
  inline double theZ() const { return 14.0000; }
  ///Density in g/cm3
  inline double rho() const { return 2.329; }
  ///One radiation length in cm
  inline double radLenIncm() const { return 9.360; }
  ///Mean excitation energy (in GeV)
  inline double excitE() const { return 12.5E-9*theZ(); }
  ///Electron mass in GeV/c2
  inline double eMass() const { return 0.000510998902; }


  /// Compute the material effect (calls the sub class)
  void updateState(ParticlePropagator& myTrack, double radlen);
  
  /// Returns const iterator to the beginning of the daughters list
  RHEP_const_iter beginDaughters() const {return _theUpdatedState.begin();};

  /// Returns const iterator to the end of the daughters list
  RHEP_const_iter endDaughters() const {return _theUpdatedState.end();};

  /// Returns the number of daughters 
  unsigned nDaughters() const {return _theUpdatedState.size();};

  /// Sets the vector normal to the surface traversed 
  void setNormalVector(const GlobalVector& normal) { 
    theNormalVector = normal;
  }

 private:

  /// Overloaded in all material effects updtators
  virtual void compute(ParticlePropagator& Particle ) = 0;

  /// Returns the fraction of radiation lengths traversed
  inline double radiationLength() const {return radLengths;}


 protected:

  mutable std::list<const RawParticle*> _theUpdatedState;

  double radLengths;

  GlobalVector theNormalVector;

  const RandomEngine* random;

};

#endif
