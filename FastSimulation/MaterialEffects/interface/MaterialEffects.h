
#ifndef MaterialEffects_h
#define MaterialEffects_h

/**
 * Steering class for the simulation of the Material Effects in each 
 * layer of the tracker material. For now, it has
 *
 *  - Photon conversion
 *  - Bremsstrahlung
 *  - Energy loss by ionization
 *  - Multiple scattering
 *  - Muon Bremsstrahlung

 * but no synchrotron radiation (well, this is not really a material 
 * effect, but might be dealt with here as well), no nuclear interactions, 
 * no delta-rays.
 *
 * The method interact() does all the above items in turn, and modifies 
 * the FSimEvent accordingly. For instance, a converting photon gets 
 * an end vertex here, and two new FSimTracks are created, one for the
 * electron and one for the positron, with the same parent vertex.
 *
 * \author: Stephan Wynhoff, Florian Beaudette, Patrick Janot
 * $Date: Last modification (after severe clean-up). 27-Fev-2011- Sandro Fonseca and Andre Sznajder  (UERJ/Brazil)
 */

//Framework Headers
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Geometry Headers
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include <vector>
class FSimEvent;
class TrackerLayer;
class ParticlePropagator;
class PairProductionSimulator;
class BremsstrahlungSimulator;
class EnergyLossSimulator;
class NuclearInteractionSimulator;
class RandomEngine;
class MultipleScatteringSimulator;
class MuonBremsstrahlungSimulator;

namespace edm {
  class ParameterSet;
}

class MaterialEffects
{

 public:

  /// Constructor
  MaterialEffects(const edm::ParameterSet& matEff,
		  const RandomEngine* engine);

  /// Default destructor
  ~MaterialEffects();

  /// Steer the various interaction processes in the Tracker Material
  /// and update the FSimEvent
  void interact(FSimEvent& simEvent,
		const TrackerLayer& layer,
		ParticlePropagator& PP,
		unsigned i);

  /// Save nuclear interaction information
  void save();

  /// Return the thickness of the current layer
  inline double thickness() const { return theThickness; }

  /// Return the energy loss by ionization in the current layer
  inline double energyLoss() const { return theEnergyLoss; }

  /// Return the Multiple Scattering engine
  inline MultipleScatteringSimulator* multipleScatteringSimulator() const { 
    return MultipleScattering;
  }

  /// Return the Energy Loss engine
  inline EnergyLossSimulator* energyLossSimulator() const { 
    return EnergyLoss;
  }

/// Return the Muon Bremsstrahlung engine
  inline MuonBremsstrahlungSimulator* muonBremsstrahlungSimulator() const {
    return MuonBremsstrahlung;
  }

 private:

  /// The number of radiation lengths traversed
  double radLengths(const TrackerLayer& layer,
		    ParticlePropagator& myTrack);

  /// The vector normal to the surface traversed
  GlobalVector normalVector(const TrackerLayer& layer,
			    ParticlePropagator& myTrack ) const;

 private:

  PairProductionSimulator* PairProduction;
  BremsstrahlungSimulator* Bremsstrahlung;
////// Muon Brem
  MuonBremsstrahlungSimulator* MuonBremsstrahlung;
  MultipleScatteringSimulator* MultipleScattering;
  EnergyLossSimulator* EnergyLoss;
  NuclearInteractionSimulator* NuclearInteraction;

  // Cuts for material effects
  double pTmin;
  GlobalVector theNormalVector;
  double theThickness;
  double theEnergyLoss;
  double theTECFudgeFactor;

  // debugging
  //  double myEta;

  // The random engine
  const RandomEngine* random;

  bool use_hardcoded;

};

#endif
