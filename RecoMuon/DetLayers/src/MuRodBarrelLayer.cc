/** \file
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */


#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

//#include "CommonReco/DetLayers/interface/GeneralBinFinderInPhi.h"
//#include "CommonReco/DetLayers/interface/PhiBorderFinder.h"

#include <algorithm>
#include <iostream>


MuRodBarrelLayer::MuRodBarrelLayer(vector<const DetRod*>& rods) :
  theRods(rods),
  isOverlapping(false) {
    // FIXME init binfinder
  }


MuRodBarrelLayer::~MuRodBarrelLayer() {}



pair<bool, TrajectoryStateOnSurface>
MuRodBarrelLayer::compatible(const TrajectoryStateOnSurface& ts, const Propagator& prop, 
			     const MeasurementEstimator& est) const {
  // FIXME
  return  make_pair(bool(), TrajectoryStateOnSurface());
}


vector<GeometricSearchDet::DetWithState> 
MuRodBarrelLayer::compatibleDets(const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop, 
				 const MeasurementEstimator& est) const {
  // FIXME
  return vector<DetWithState>();
}


vector<DetGroup> 
MuRodBarrelLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
					 const Propagator& prop,
					 const MeasurementEstimator& est) const {
  // FIXME
  return vector<DetGroup>();
}


bool MuRodBarrelLayer::hasGroups() const {
  // FIXME : depending on isOverlapping?
  return false;
}


Module MuRodBarrelLayer::module() {
  // FIXME
  return Module();
}
