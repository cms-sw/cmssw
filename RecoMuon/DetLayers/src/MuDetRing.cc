/** \file
 *
 *  $Date: 2012/05/29 08:23:54 $
 *  $Revision: 1.9 $
 *  \author N. Amapane - CERN
 */

#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <vector>

using namespace std;

MuDetRing::MuDetRing(vector<const GeomDet*>::const_iterator first,
		     vector<const GeomDet*>::const_iterator last) : 
  ForwardDetRingOneZ(first,last) 
{
  init();
}


MuDetRing::MuDetRing(const vector<const GeomDet*>& vdets) : 
  ForwardDetRingOneZ(vdets) 
{
  init();
}


void MuDetRing::init()
{
  theBinFinder = BinFinderType(basicComponents().front()->position().phi(),
			       basicComponents().size());  
}

MuDetRing::~MuDetRing(){}


const vector<const GeometricSearchDet*>&
MuDetRing::components() const {
  // FIXME dummy impl.
  cout << "temporary dummy implementation of MuDetRing::components()!!" << endl;
  static vector<const GeometricSearchDet*> result;
  return result;
}


pair<bool, TrajectoryStateOnSurface>
MuDetRing::compatible(const TrajectoryStateOnSurface& ts, const Propagator& prop, 
		      const MeasurementEstimator& est) const {
  
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuDetRing";
  TrajectoryStateOnSurface ms = prop.propagate(ts,specificSurface());
  
  LogTrace(metname) << "MuDetRing::compatible, Surface at Z: " 
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
MuDetRing::compatibleDets( const TrajectoryStateOnSurface& startingState,
			   const Propagator& prop, 
			   const MeasurementEstimator& est) const {

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuDetRing";

  LogTrace(metname) << "MuDetRing::compatibleDets, Surface at Z: " 
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
    LogTrace(metname) << "    MuDetRing::compatibleDets: not compatible"
                      << "    (should not have been selected!)";
    return result;
  }
  
  // Find the most probable destination component
  TrajectoryStateOnSurface& tsos = compat.second;
  GlobalPoint startPos = tsos.globalPosition();  
  int closest = theBinFinder.binIndex(startPos.phi());
  const vector<const GeomDet*> dets = basicComponents();
  LogTrace(metname) << "     MuDetRing::compatibleDets, closest det: " << closest 
                    << " Phi: " << dets[closest]->surface().position().phi()
                    << " impactPhi " << startPos.phi();

  // Add this detector, if it is compatible
  // NOTE: add performs a null propagation
  add(closest, result, tsos, prop, est);

  int nclosest = result.size(); int nnextdet=0; // MDEBUG counters

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
    LogTrace(metname) << "     MuDetRing::compatibleDets, closest not compatible!";
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
      nnextdet++;      
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
      nnextdet++;
      if ( !add(idetp, result, tsos, prop, est)) break;
    }
  }
  
  LogTrace(metname) << "     MuDetRing::compatibleDets, size: " << result.size()
                    << " on closest: " << nclosest << " # checked dets: " << nnextdet+1;

  if (result.size()==0) {
    LogTrace(metname) << "   ***Ring not compatible,should have been discarded before!!!";
  }
  
  return result;
}


vector<DetGroup> 
MuDetRing::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
				  const Propagator& prop,
				  const MeasurementEstimator& est) const {
  // FIXME should be implemented to allow returning  overlapping chambers
  // as separate groups!
  cout << "dummy implementation of MuDetRod::groupedCompatibleDets()" << endl;
  vector<DetGroup> result;
  return result;
}
