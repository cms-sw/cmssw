/** \file
 *
 *  \author L. Gray - FNAL
 */

#include "RecoMTD/DetLayers/interface/MTDDetRing.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>

using namespace std;

MTDDetRing::MTDDetRing(vector<const GeomDet*>::const_iterator first,
		     vector<const GeomDet*>::const_iterator last) : 
  ForwardDetRingOneZ(first,last) 
{
  init();
}


MTDDetRing::MTDDetRing(const vector<const GeomDet*>& vdets) : 
  ForwardDetRingOneZ(vdets) 
{
  init();
}


void MTDDetRing::init()
{
  theBinFinder = BinFinderType(basicComponents().front()->position().phi(),
			       basicComponents().size());  
}

MTDDetRing::~MTDDetRing(){}


const vector<const GeometricSearchDet*>&
MTDDetRing::components() const {
  // FIXME dummy impl.
  edm::LogError("MTDDetRing") << "temporary dummy implementation of MTDDetRing::components()!!" << endl;
  static const vector<const GeometricSearchDet*> result;
  return result;
}


pair<bool, TrajectoryStateOnSurface>
MTDDetRing::compatible(const TrajectoryStateOnSurface& ts, const Propagator& prop, 
		      const MeasurementEstimator& est) const {
  
  const std::string metname = "MTD|RecoMTD|RecoMTDDetLayers|MTDDetRing";
  TrajectoryStateOnSurface ms = prop.propagate(ts,specificSurface());
  
  LogTrace(metname) << "MTDDetRing::compatible, Surface at Z: " 
                    << specificSurface().position().z()
                    << " R1: " << specificSurface().innerRadius()
                    << " R2: " << specificSurface().outerRadius()
                    << " TS   at Z,R: " << ts.globalPosition().z() << ","
                    << ts.globalPosition().perp();
  if (ms.isValid()) {
    LogTrace(metname) << " DEST at Z,R: " << ms.globalPosition().z() << ","
		      << ms.globalPosition().perp()
		      << " local Z: "   << ms.localPosition().z()  << endl;
  }
  else
    LogTrace(metname) << " DEST: not valid" <<endl;
  
  
  if (ms.isValid()) return make_pair(est.estimate(ms, specificSurface()) != 0, ms);
  else return make_pair(false, ms);
}


vector<GeometricSearchDet::DetWithState> 
MTDDetRing::compatibleDets( const TrajectoryStateOnSurface& startingState,
			   const Propagator& prop, 
			   const MeasurementEstimator& est) const {

  const std::string metname = "MTD|RecoMTD|RecoMTDDetLayers|MTDDetRing";

  LogTrace(metname) << "MTDDetRing::compatibleDets, Surface at Z: " 
                    << surface().position().z()
                    << " R1: " << specificSurface().innerRadius()
                    << " R2: " << specificSurface().outerRadius()
                    << " TS at Z,R: " << startingState.globalPosition().z() << ","
                    << startingState.globalPosition().perp() << "     DetRing pos." << position();

  vector<DetWithState> result;

  // Propagate and check that the result is within bounds
  pair<bool, TrajectoryStateOnSurface> compat =
    compatible(startingState, prop, est);
  if (!compat.first) {
    LogTrace(metname) << "    MTDDetRing::compatibleDets: not compatible"
                      << "    (should not have been selected!)";
    return result;
  }
  
  // Find the most probable destination component
  TrajectoryStateOnSurface& tsos = compat.second;
  GlobalPoint startPos = tsos.globalPosition();  
  int closest = theBinFinder.binIndex(startPos.phi());
  const vector<const GeomDet*> dets = basicComponents();
  LogTrace(metname) << "     MTDDetRing::compatibleDets, closest det: " << closest 
                    << " Phi: " << dets[closest]->surface().position().phi()
                    << " impactPhi " << startPos.phi();

  // Add this detector, if it is compatible
  // NOTE: add performs a null propagation
  add(closest, result, tsos, prop, est);

#ifdef EDM_ML_DEBUG
  int nclosest = result.size(); int nnextdet=0; // MDEBUG counters
#endif

  // Try the neighbors on each side until no more compatible.
  float dphi=0;
  if (!result.empty()) { // If closest is not compatible the next cannot be either
    float nSigmas = 3.;
    if (result.back().second.hasError()) {
      dphi = nSigmas*      
        atan(sqrt(result.back().second.localError().positionError().xx())/
             result.back().second.globalPosition().perp());
    }  
  } else {
    LogTrace(metname) << "     MTDDetRing::compatibleDets, closest not compatible!";
    //FIXME:  if closest is not compatible the next cannot be either
  }
  
  for (int idet=closest+1; idet < closest+int(dets.size())/4+1; idet++){
    // FIXME: should use dphi to decide if det must be queried.
    // Right now query until not compatible.
    int idetp = theBinFinder.binIndex(idet);
    {
      LogTrace(metname) << "     next det:" << idetp
                        << " at Z: " << dets[idetp]->position().z()
                        << " phi: " << dets[idetp]->position().phi()
                        << " FTS phi " << startPos.phi()
                        << " max dphi " << dphi;
#ifdef EDM_ML_DEBUG
      nnextdet++;  
#endif
      if ( !add(idetp, result, tsos, prop, est)) break;
    }
  }
  
  for (int idet=closest-1; idet > closest-int(dets.size())/4-1; idet--){
    // FIXME: should use dphi to decide if det must be queried.
    // Right now query until not compatible.
    int idetp = theBinFinder.binIndex(idet);
    {
      LogTrace(metname) << "     previous det:" << idetp << " " << idet << " " << closest-dets.size()/4-1
                        << " at Z: " << dets[idetp]->position().z()
                        << " phi: " << dets[idetp]->position().phi()
                        << " FTS phi " << startPos.phi()
                        << " max dphi" << dphi;
#ifdef EDM_ML_DEBUG
      nnextdet++;
#endif
      if ( !add(idetp, result, tsos, prop, est)) break;
    }
  }
  
#ifdef EDM_ML_DEBUG
  LogTrace(metname) << "     MTDDetRing::compatibleDets, size: " << result.size()
                    << " on closest: " << nclosest << " # checked dets: " << nnextdet+1;
#endif

  if (result.empty()) {
    LogTrace(metname) << "   ***Ring not compatible,should have been discarded before!!!";
  }
  
  return result;
}


vector<DetGroup> 
MTDDetRing::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
				  const Propagator& prop,
				  const MeasurementEstimator& est) const {
  // FIXME should be implemented to allow returning  overlapping chambers
  // as separate groups!
  cout << "dummy implementation of MTDDetRod::groupedCompatibleDets()" << endl;
  vector<DetGroup> result;
  return result;
}
