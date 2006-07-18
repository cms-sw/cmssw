/** \file
 *
 *  $Date: 2006/06/27 12:33:29 $
 *  $Revision: 1.11 $
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

using namespace std;

#define MDEBUG false //FIXME!

MuRingForwardLayer::MuRingForwardLayer(vector<const ForwardDetRing*>& rings) :
  theRings(rings),
  theComponents(theRings.begin(),theRings.end()),
  theBinFinder(0),
  isOverlapping(false) 
{
  // Cache chamber pointers (the basic components_)
  for (vector<const ForwardDetRing*>::const_iterator it=rings.begin();
       it!=rings.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(),tmp2.begin(),tmp2.end());
  }  

  RBorderFinder bf(theRings);
  isOverlapping = bf.isROverlapping();
  theBinFinder = new GeneralBinFinderInR<double>(bf);

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
  delete theBinFinder;
  for (vector <const ForwardDetRing*>::iterator i = theRings.begin();
       i<theRings.end(); i++) {delete *i;}
}


vector<GeometricSearchDet::DetWithState> 
MuRingForwardLayer::compatibleDets(const TrajectoryStateOnSurface& startingState,
				   const Propagator& prop, 
				   const MeasurementEstimator& est) const {
  vector<DetWithState> result; 
  // FIXME
  if ( MDEBUG ) 
    cout << "MuRingForwardLayer::compatibleDets," 
 	 << " R1 " << specificSurface().innerRadius()
	 << " R2: " << specificSurface().outerRadius()
 	 << " FTS at R: " << startingState.globalPosition().perp()
 	 << endl;

  pair<bool, TrajectoryStateOnSurface> compat =
    compatible(startingState, prop, est);

  if (!compat.first) {
    if ( MDEBUG )
      cout << "     MuRingForwardLayer::compatibleDets: not compatible"
	   << " (should not have been selected!)" <<endl;
    return result;
  }


  TrajectoryStateOnSurface& tsos = compat.second;
  
  int closest = theBinFinder->binIndex(tsos.globalPosition().perp());
  const ForwardDetRing* closestRing = theRings[closest];

  // Check the closest ring
  if ( MDEBUG ) {
    cout << "     MuRingForwardLayer::fastCompatibleDets, closestRing: "
 	 << closest
 	 << " R1 " << closestRing->specificSurface().innerRadius()
	 << " R2: " << closestRing->specificSurface().outerRadius()
 	 << " FTS R: " << tsos.globalPosition().perp();
    if (tsos.hasError()) {
      cout << " sR: " << sqrt(tsos.localError().positionError().yy())
	   << " sX: " << sqrt(tsos.localError().positionError().xx());
    }
    cout << endl;
  }

  result = closestRing->compatibleDets(tsos, prop, est);

  int nclosest = result.size(); int nnextdet=0; // MDEBUG counters

  //FIXME: if closest is not compatible next cannot be either?
  
  // Use state on layer surface. Note that local coordinates and errors
  // are the same on the layer and on all rings surfaces, since 
  // all BoundDisks are centered in 0,0 and have the same rotation.
  // CAVEAT: if the rings are not at the same Z, the local position and error
  // will be "Z-projected" to the rings. This is a fairly good approximation.
  // However in this case additional propagation will be done when calling
  // compatibleDets.
  GlobalPoint startPos = tsos.globalPosition();
  LocalPoint nextPos(surface().toLocal(startPos));
  
  for (unsigned int idet=closest+1; idet < theRings.size(); idet++) {
    bool inside = false;
    if (tsos.hasError()) {
      inside=theRings[idet]->specificSurface().bounds().inside(nextPos,tsos.localError().positionError());
    } else {
      inside=theRings[idet]->specificSurface().bounds().inside(nextPos);
    }
    if (inside){
      if ( MDEBUG ) 
	cout << "     MuRingForwardLayer::fastCompatibleDets:NextRing" << idet
	     << " R1 " << theRings[idet]->specificSurface().innerRadius()
	     << " R2: " << theRings[idet]->specificSurface().outerRadius()
	     << " FTS R " << nextPos.perp()
	     << endl;
      nnextdet++;      
      vector<DetWithState> nextRodDets =
	theRings[idet]->compatibleDets(tsos, prop, est);
      if (nextRodDets.size()!=0) {
	result.insert( result.end(), 
		       nextRodDets.begin(), nextRodDets.end());
      } else {
	break;
      }
    }
  }

  for (int idet=closest-1; idet >= 0; idet--) {
    bool inside = false;
    if (tsos.hasError()) {
      inside=theRings[idet]->specificSurface().bounds().inside(nextPos,tsos.localError().positionError());
    } else {
      inside=theRings[idet]->specificSurface().bounds().inside(nextPos);
    }
    if (inside){
      if ( MDEBUG ) 
	cout << "     MuRingForwardLayer::fastCompatibleDets:PreviousRing:" << idet
	     << " R1 " << theRings[idet]->specificSurface().innerRadius()
	     << " R2: " << theRings[idet]->specificSurface().outerRadius()
	     << " FTS R " << nextPos.perp()
	     << endl;
      nnextdet++;
      vector<DetWithState> nextRodDets =
	theRings[idet]->compatibleDets(tsos, prop, est);
      if (nextRodDets.size()!=0) {
	result.insert( result.end(), 
		       nextRodDets.begin(), nextRodDets.end());
      } else {
	break;
      }
    }
  }
  
  if ( MDEBUG ) 
    cout << "     MuRingForwardLayer::fastCompatibleDets: found: "
	 << result.size()
	 << " on closest: " << nclosest
	 << " # checked rings: " << 1 + nnextdet
	 << endl;
  
  return result;
}


vector<DetGroup> 
MuRingForwardLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
					   const Propagator& prop,
					   const MeasurementEstimator& est) const {
  // FIXME should return only 1 group 
  cout << "dummy implementation of MuRingForwardLayer::groupedCompatibleDets()" << endl;
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

const vector<const GeometricSearchDet*> &
MuRingForwardLayer::components() const {
  return theComponents;
}
