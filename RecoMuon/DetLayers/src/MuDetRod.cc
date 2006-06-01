/** \file
 *
 *  $Date: 2006/05/16 09:43:00 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include <iostream>

#define MDEBUG false //FIXME!

MuDetRod::MuDetRod(vector<const GeomDet*>::const_iterator first,
                   vector<const GeomDet*>::const_iterator last)
  : DetRodOneR(first,last) {
    init();
  }

MuDetRod::MuDetRod(const vector<const GeomDet*>& vdets)
  : DetRodOneR(vdets) {
    init();
  }


void MuDetRod::init() {
  theBinFinder = BinFinderType(basicComponents().begin(), basicComponents().end());
}


MuDetRod::~MuDetRod(){}


const vector<const GeometricSearchDet*>&
MuDetRod::components() const {
  // FIXME dummy impl.
  cout << "temporary dummy implementation of MuDetRod::components()!!" << endl;
  static vector<const GeometricSearchDet*> result;
  return result;
}


pair<bool, TrajectoryStateOnSurface>
MuDetRod::compatible(const TrajectoryStateOnSurface& ts, const Propagator& prop, 
		     const MeasurementEstimator& est) const {

  TrajectoryStateOnSurface ms = prop.propagate(ts,specificSurface());
  if (ms.isValid()) return make_pair(est.estimate(ms, specificSurface()) != 0, ms);
  else return make_pair(false, ms);
}


vector<GeometricSearchDet::DetWithState> 
MuDetRod::compatibleDets( const TrajectoryStateOnSurface& startingState,
			  const Propagator& prop, 
			  const MeasurementEstimator& est) const {

  if ( MDEBUG ) cout << "MuDetRod::compatibleDets, Surface at R,phi: " 
		    << surface().position().perp()  << ","
		    << surface().position().phi() << endl
// FIXME	    << " TS at R,phi: " << startingState.position().perp() << ","
// 		    << startingState.position().phi()
		    << endl
		    << "     DetRod pos." << position() 
		    << endl;
  

  vector<DetWithState> result;

  // Propagate and check that the result is within bounds
  pair<bool, TrajectoryStateOnSurface> compat =
    compatible(startingState, prop, est);
  
  if (!compat.first) {
    if ( MDEBUG ) cout << "    MuDetRod::compatibleDets: not compatible"
		      << "    (should not have been selected!)" <<endl;
    return result;
  }

  // Find the most probable destination component
  TrajectoryStateOnSurface& tsos = compat.second;
  GlobalPoint startPos = tsos.globalPosition();
  int closest = theBinFinder.binIndex(startPos.z());
  const vector<const GeomDet*> dets = basicComponents();
  if ( MDEBUG ) cout << "     MuDetRod::compatibleDets, closest det: " << closest 
		    << " pos: " << dets[closest]->surface().position()
		    << " impact " << startPos
		    << endl;

  // Add this detector, if it is compatible
  // NOTE: add performs a null propagation
  add(closest, result, tsos, prop, est);

  int nclosest = result.size(); int nnextdet=0; // just DEBUG counters

  // Try the neighbors on each side until no more compatible.
  // If closest is not compatible the next cannot be either
  if (!result.empty()) {
    const BoundPlane& closestPlane(dynamic_cast<const BoundPlane&>
				    (dets[closest]->surface()));    
     MeasurementEstimator::Local2DVector maxDistance = 
      est.maximalLocalDisplacement( result.front().second, closestPlane);

    // detHalfLen is assumed to be the same for all detectors.
    float detHalfLen = closestPlane.bounds().length()/2.;

    for (unsigned int idet=closest+1; idet < dets.size(); idet++) {
      LocalPoint nextPos(dets[idet]->toLocal(startPos));
      if (fabs(nextPos.y()) < detHalfLen + maxDistance.y()) { 
        if ( MDEBUG ) cout << "     negativeZ: det:" << idet
			  << " pos " << nextPos.y()
			  << " maxDistance " << maxDistance.y()
			  << endl;
        nnextdet++;
        if ( !add(idet, result, tsos, prop, est)) break;
      } else {
	break;
      }
    }

    for (int idet=closest-1; idet >= 0; idet--) {
      LocalPoint nextPos( dets[idet]->toLocal(startPos));
      if (fabs(nextPos.y()) < detHalfLen + maxDistance.y()) {
	if ( MDEBUG ) cout << "     positiveZ: det:" << idet
			  << " pos " << nextPos.y()
			  << " maxDistance " << maxDistance.y()
			  << endl;
        nnextdet++;
        if ( !add(idet, result, tsos, prop, est)) break;
      } else {
	break;
      }
    }
  }

  if ( MDEBUG ) cout << "     MuDetRod::compatibleDets, size: " << result.size()
		    << " on closest: " << nclosest
		    << " # checked dets: " << nnextdet+1
		    << endl;
  if (result.size()==0) {
    if ( MDEBUG ) cout << "   ***Rod not compatible---should have been discarded before!!!" <<endl;
  }
  return result;
}


vector<DetGroup> 
MuDetRod::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop,
				 const MeasurementEstimator& est) const {
  // FIXME should return only 1 group 
  cout << "dummy implementation of MuDetRod::groupedCompatibleDets()" << endl;
  vector<DetGroup> result;
  return result;
}
