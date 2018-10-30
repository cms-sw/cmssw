/** \file
 *
 *  \author L. Gray - FNAL
 */

#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

using namespace std;


MTDDetTray::MTDDetTray(vector<const GeomDet*>::const_iterator first,
                   vector<const GeomDet*>::const_iterator last)
  : DetRodOneR(first,last) {
  init();
}

MTDDetTray::MTDDetTray(const vector<const GeomDet*>& vdets)
  : DetRodOneR(vdets) {
  init();
}


void MTDDetTray::init() {
  theBinFinder = BinFinderType(basicComponents().begin(), basicComponents().end());
}


MTDDetTray::~MTDDetTray(){}

const vector<const GeometricSearchDet*>&
MTDDetTray::components() const {

  // FIXME dummy impl.
  edm::LogError("MTDDetTray") << "temporary dummy implementation of MTDDetTray::components()!!" << endl;
  static const vector<const GeometricSearchDet*> result;
  return result;
}

pair<bool, TrajectoryStateOnSurface>
MTDDetTray::compatible(const TrajectoryStateOnSurface& ts, const Propagator& prop, 
                     const MeasurementEstimator& est) const {
  
  TrajectoryStateOnSurface ms = prop.propagate(ts,specificSurface());
  if (ms.isValid()) return make_pair(est.estimate(ms, specificSurface()) != 0, ms);
  else return make_pair(false, ms);
}


vector<GeometricSearchDet::DetWithState> 
MTDDetTray::compatibleDets( const TrajectoryStateOnSurface& startingState,
                          const Propagator& prop, 
                          const MeasurementEstimator& est) const {
  const std::string metname = "MTD|RecoMTD|RecoMTDDetLayers|MTDDetTray";
  
  LogTrace(metname) << "MTDDetTray::compatibleDets, Surface at R,phi: " 
                    << surface().position().perp()  << ","
                    << surface().position().phi() << "     DetRod pos.";
    // FIXME	    << " TS at R,phi: " << startingState.position().perp() << ","
    // 		    << startingState.position().phi()

    
  vector<DetWithState> result;
  
  // Propagate and check that the result is within bounds
  pair<bool, TrajectoryStateOnSurface> compat =
    compatible(startingState, prop, est);
  
  if (!compat.first) {
    LogTrace(metname) << "    MTDDetTray::compatibleDets: not compatible"
                      << "    (should not have been selected!)";
    return result;
  }
  
  // Find the most probable destination component
  TrajectoryStateOnSurface& tsos = compat.second;
  GlobalPoint startPos = tsos.globalPosition();
  int closest = theBinFinder.binIndex(startPos.z());
  const vector<const GeomDet*> dets = basicComponents();
  LogTrace(metname) << "     MTDDetTray::compatibleDets, closest det: " << closest 
                    << " pos: " << dets[closest]->surface().position()
                    << " impact " << startPos;
  
  // Add this detector, if it is compatible
  // NOTE: add performs a null propagation
  add(closest, result, tsos, prop, est);
  
#ifdef EDM_ML_DEBUG
  int nclosest = result.size(); int nnextdet=0; // just DEBUG counters
#endif
  
  // Try the neighbors on each side until no more compatible.
  // If closest is not compatible the next cannot be either
  if (!result.empty()) {
    const BoundPlane& closestPlane(dets[closest]->surface());
    MeasurementEstimator::Local2DVector maxDistance = 
      est.maximalLocalDisplacement( result.front().second, closestPlane);
    
    // detHalfLen is assumed to be the same for all detectors.
    float detHalfLen = closestPlane.bounds().length()/2.;
    
    for (unsigned int idet=closest+1; idet < dets.size(); idet++) {
      LocalPoint nextPos(dets[idet]->toLocal(startPos));
      if (fabs(nextPos.y()) < detHalfLen + maxDistance.y()) { 
        LogTrace(metname) << "     negativeZ: det:" << idet
                          << " pos " << nextPos.y()
                          << " maxDistance " << maxDistance.y();
#ifdef EDM_ML_DEBUG
        nnextdet++;
#endif
        if ( !add(idet, result, tsos, prop, est)) break;
      } else {
        break;
      }
    }
    
    for (int idet=closest-1; idet >= 0; idet--) {
      LocalPoint nextPos( dets[idet]->toLocal(startPos));
      if (fabs(nextPos.y()) < detHalfLen + maxDistance.y()) {
        LogTrace(metname) << "     positiveZ: det:" << idet
                          << " pos " << nextPos.y()
                          << " maxDistance " << maxDistance.y();
#ifdef EDM_ML_DEBUG
        nnextdet++;
#endif
        if ( !add(idet, result, tsos, prop, est)) break;
      } else {
        break;
      }
    }
  }
  
#ifdef EDM_ML_DEBUG
  LogTrace(metname) << "     MTDDetTray::compatibleDets, size: " << result.size()
                    << " on closest: " << nclosest
                    << " # checked dets: " << nnextdet+1;
#endif
  if (result.empty()) {
    LogTrace(metname) << "   ***Rod not compatible---should have been discarded before!!!";
  }
  return result;
}


vector<DetGroup> 
MTDDetTray::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
                                 const Propagator& prop,
                                 const MeasurementEstimator& est) const {
  // FIXME should return only 1 group 
  cout << "dummy implementation of MTDDetTray::groupedCompatibleDets()" << endl;
  vector<DetGroup> result;
  return result;
}
