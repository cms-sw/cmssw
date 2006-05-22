/** \file
 *
 *  $Date: 2006/05/18 14:52:41 $
 *  $Revision: 1.5 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <TrackingTools/GeomPropagators/interface/Propagator.h>
#include <TrackingTools/PatternTools/interface/MeasurementEstimator.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "RBorderFinder.h"
#include "GeneralBinFinderInR.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define MDEBUG false //FIXME!

MuRingForwardLayer::MuRingForwardLayer(vector<const ForwardDetRing*>& rings) :
  theRings(rings),
  isOverlapping(false) 
{
  // Cache chamber pointers (the basic components_)
  for (vector<const ForwardDetRing*>::const_iterator it=rings.begin();
       it!=rings.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(),tmp2.begin(),tmp2.end());
  }  

//   RBorderFinder bf(rings); // FIXME: change the iface of RBorderFinder...
//   isOverlapping = bf.isROverlapping();
//   theBinFinder = new GeneralBinFinderInR<double>(bf);

  ForwardDetLayer::initialize(); // Compute surface

  if ( MDEBUG ) 
    cout << "Constructing MuRingForwardLayer: "
	 << basicComponents().size() << " Dets " 
	 << theRings.size() << " Rings "
	 << " Z: " << specificSurface().position().z()
// 	 << " Per.: " << bf.isRPeriodic()
// 	 << " Overl.: " << isOverlapping
	 << endl;
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
  // FIXME! will be used also for RPC
  return csc;
}

vector<const GeometricSearchDet*> 
MuRingForwardLayer::components() const {
  vector <const GeometricSearchDet*> result(theRings.begin(),theRings.end());
  return result;
}
