#ifndef MATERIALEFFECTSSIMULATOR_H
#define MATERIALEFFECTSSIMULATOR_H

#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

#include <vector>

class RandomEngineAndDistribution;

/** 
 * This is the generic class for Material Effects in the tracker material, 
 * from which FamosPairProductionSimulator, FamosBremsstrahlungSimulator, 
 * FamosEnergyLossSimulator and FamosMultipleScatteringSimulator inherit. It
 * determines the fraction of radiation lengths traversed by the particle 
 * in this tracker layer, defines the tracker layer characteristics (hmmm,
 * 100% Silicon, very crude, but what can we do?) and returns a list of
 * new RawParticles if needed.
 *
 * \author Stephan Wynhoff, Florian Beaudette, Patrick Janot
 * $Date: Last update 8-Jan-2004
 */ 

class MaterialEffectsSimulator
{
 public:

  typedef std::vector<RawParticle>::const_iterator RHEP_const_iter;

  // Constructor : default values are for Silicon
  MaterialEffectsSimulator(double A = 28.0855,
			   double Z = 14.0000,
			   double density = 2.329,
			   double radLen = 9.360);

  virtual ~MaterialEffectsSimulator();

  /// Functions to return atomic properties of the material
  /// Here the tracker material is assumed to be 100% Silicon

  /// A
  inline double theA() const { return A; }
  /// Z
  inline double theZ() const { return Z; }
  ///Density in g/cm3
  inline double rho() const { return density; }
  ///One radiation length in cm
  inline double radLenIncm() const { return radLen; }
  ///Mean excitation energy (in GeV)
  inline double excitE() const { return 12.5E-9*theZ(); }
  ///Electron mass in GeV/c2
  inline double eMass() const { return 0.000510998902; }

  /// Compute the material effect (calls the sub class)
  void updateState(ParticlePropagator& myTrack, double radlen, RandomEngineAndDistribution const*);
  
  /// Returns const iterator to the beginning of the daughters list
  inline RHEP_const_iter beginDaughters() const {return _theUpdatedState.begin();}

  /// Returns const iterator to the end of the daughters list
  inline RHEP_const_iter endDaughters() const {return _theUpdatedState.end();}

  /// Returns the number of daughters 
  inline unsigned nDaughters() const {return _theUpdatedState.size();}

  /// Sets the vector normal to the surface traversed 
  inline void setNormalVector(const GlobalVector& normal) { theNormalVector = normal; }
  
  /// A vector orthogonal to another one (because it's not in XYZTLorentzVector)
  XYZVector orthogonal(const XYZVector&) const; 

  /// The id of the closest charged daughter (filled for nuclear interactions only)
  inline int closestDaughterId() { return theClosestChargedDaughterId; } 

  /// Used by  NuclearInteractionSimulator to save last sampled event
  virtual void save() {};

 private:

  /// Overloaded in all material effects updtators
  virtual void compute(ParticlePropagator& Particle, RandomEngineAndDistribution const*) = 0;

  /// Returns the fraction of radiation lengths traversed
  inline double radiationLength() const {return radLengths;}

 protected:

  std::vector<RawParticle> _theUpdatedState;

  double radLengths;

  // Material properties
  double A;
  double Z;
  double density;
  double radLen;

  GlobalVector theNormalVector;

  int theClosestChargedDaughterId;

};

#endif
