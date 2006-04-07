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
 *
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
 * $Date: Last modification (after severe clean-up). 08-Jan-2004
 */

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

// Famos Headers
#include "FastSimulation/MaterialEffects/interface/PairProductionUpdator.h"
#include "FastSimulation/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "FastSimulation/MaterialEffects/interface/BremsstrahlungUpdator.h"
#include "FastSimulation/MaterialEffects/interface/EnergyLossUpdator.h"



#include <iostream>

class TrackerLayer;
class ParticlePropagator;

class MaterialEffects
{

 public:

  /// Constructor
  MaterialEffects();

  /// Default destructor
  ~MaterialEffects(){}

  /// Steer the various interaction processes in the Tracker Material
  /// and update the FSimEvent
  void interact(FSimEvent& simEvent,
		const TrackerLayer& layer,
		ParticlePropagator& PP,
		unsigned i);

  /// The number of radiation lengths traversed
  double radLengths(const TrackerLayer& layer,
		    ParticlePropagator& myTrack ) const;

  /// The vector normal to the surface traversed
  GlobalVector normalVector(const TrackerLayer& layer,
			    ParticlePropagator& myTrack ) const;

 private:

  PairProductionUpdator PairProduction;
  BremsstrahlungUpdator Bremsstrahlung;
  MultipleScatteringUpdator MultipleScattering;
  EnergyLossUpdator EnergyLoss;


  bool doPairProduction;
  bool doBremsstrahlung;
  bool doEnergyLoss;
  bool doMultipleScattering;

  double pTmin;

  // debugging
  double myEta;

};

#endif
