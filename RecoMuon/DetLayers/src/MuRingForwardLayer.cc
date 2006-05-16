/** \file
 *
 *  $Date: 2006/04/25 17:03:23 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/DetLayers/interface/MuRingForwardLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

//#include "CommonReco/DetLayers/interface/RBorderFinder.h"
//#include "CommonReco/DetLayers/interface/GeneralBinFinderInR.h"

#include <algorithm>
#include <iostream>
#include <vector>


MuRingForwardLayer::MuRingForwardLayer(vector<const ForwardDetRing*>& rings) :
  theRings(rings),
  isOverlapping(false) {
    // FIXME init binfinder    
  }


MuRingForwardLayer::~MuRingForwardLayer(){
}



pair<bool, TrajectoryStateOnSurface>
MuRingForwardLayer::compatible(const TrajectoryStateOnSurface& ts,
			       const Propagator& prop, 
			       const MeasurementEstimator& est) const {
  // FIXME
  return make_pair(bool(), TrajectoryStateOnSurface());
}


vector<GeometricSearchDet::DetWithState> 
MuRingForwardLayer::compatibleDets(const TrajectoryStateOnSurface& startingState,
				   const Propagator& prop, 
				   const MeasurementEstimator& est) const {
  // FIXME
  return vector<DetWithState>();
}


vector<DetGroup> 
MuRingForwardLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
					   const Propagator& prop,
					   const MeasurementEstimator& est) const {
  // FIXME
  return vector<DetGroup>();
}


bool MuRingForwardLayer::hasGroups() const {
  // FIXME : depending on isOverlapping?
  return false;
}


Module MuRingForwardLayer::module() const {
  // FIXME
  return Module();
}

vector<const GeometricSearchDet*> 
MuRingForwardLayer::components() const {
  return vector <const GeometricSearchDet*>(theRings.begin(),theRings.end());
}
