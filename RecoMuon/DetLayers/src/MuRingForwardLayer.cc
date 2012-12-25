/** \file
 *
 *  $Date: 2012/05/29 08:23:54 $
 *  $Revision: 1.20 $
 *  \author N. Amapane - CERN
 */

#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <DataFormats/GeometrySurface/interface/SimpleDiskBounds.h>
#include <TrackingTools/GeomPropagators/interface/Propagator.h>
#include <TrackingTools/DetLayers/interface/MeasurementEstimator.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "RBorderFinder.h"
#include "GeneralBinFinderInR.h"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

MuRingForwardLayer::MuRingForwardLayer(const vector<const ForwardDetRing*>& rings) :
  theRings(rings),
  theComponents(theRings.begin(),theRings.end()),
  theBinFinder(0),
  isOverlapping(false) 
{

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuRingForwardLayer";

  // Initial values for R and Z bounds
  float theRmin = rings.front()->basicComponents().front()->position().perp(); 
  float theRmax = theRmin;
  float theZmin = rings.front()->position().z();
  float theZmax = theZmin;

  // Cache chamber pointers (the basic components_)
  // and find extension in R and Z
  for (vector<const ForwardDetRing*>::const_iterator it=rings.begin();
       it!=rings.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(),tmp2.begin(),tmp2.end());
    
    theRmin = min( theRmin, (*it)->specificSurface().innerRadius());
    theRmax = max( theRmax, (*it)->specificSurface().outerRadius());
    float halfThick = (*it)->surface().bounds().thickness()/2.;
    float zCenter = (*it)->surface().position().z();
    theZmin = min( theZmin, zCenter-halfThick);
    theZmax = max( theZmax, zCenter+halfThick); 
  }  
  
  RBorderFinder bf(theRings);
  isOverlapping = bf.isROverlapping();
  theBinFinder = new GeneralBinFinderInR<double>(bf);

  // Build surface
  
  float zPos = (theZmax+theZmin)/2.;
  PositionType pos(0.,0.,zPos);
  RotationType rot;

  setSurface(new BoundDisk( pos, rot, 
			    new SimpleDiskBounds( theRmin, theRmax, 
					          theZmin-zPos, theZmax-zPos)));


   
  LogTrace(metname) << "Constructing MuRingForwardLayer: "
                    << basicComponents().size() << " Dets " 
                    << theRings.size() << " Rings "
                    << " Z: " << specificSurface().position().z()
                    << " R1: " << specificSurface().innerRadius()
                    << " R2: " << specificSurface().outerRadius()
                    << " Per.: " << bf.isRPeriodic()
                    << " Overl.: " << bf.isROverlapping();
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
  
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuRingForwardLayer";
  vector<DetWithState> result; 
  
  
  LogTrace(metname) << "MuRingForwardLayer::compatibleDets," 
                    << " R1 " << specificSurface().innerRadius()
                    << " R2: " << specificSurface().outerRadius()
                    << " FTS at R: " << startingState.globalPosition().perp();
  
  pair<bool, TrajectoryStateOnSurface> compat =
    compatible(startingState, prop, est);
  
  if (!compat.first) {
    
    LogTrace(metname) << "     MuRingForwardLayer::compatibleDets: not compatible"
                      << " (should not have been selected!)";
    return result;
  }
  
  
  TrajectoryStateOnSurface& tsos = compat.second;
  
  int closest = theBinFinder->binIndex(tsos.globalPosition().perp());
  const ForwardDetRing* closestRing = theRings[closest];
  
  // Check the closest ring
  
  LogTrace(metname) << "     MuRingForwardLayer::fastCompatibleDets, closestRing: "
		    << closest
		    << " R1 " << closestRing->specificSurface().innerRadius()
		    << " R2: " << closestRing->specificSurface().outerRadius()
		    << " FTS R: " << tsos.globalPosition().perp();
  if (tsos.hasError()) {
    LogTrace(metname)  << " sR: " << sqrt(tsos.localError().positionError().yy())
		       << " sX: " << sqrt(tsos.localError().positionError().xx());
  }
  LogTrace(metname) << endl;
   
   
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
      LogTrace(metname) << "     MuRingForwardLayer::fastCompatibleDets:NextRing" << idet
			<< " R1 " << theRings[idet]->specificSurface().innerRadius()
			<< " R2: " << theRings[idet]->specificSurface().outerRadius()
			<< " FTS R " << nextPos.perp();
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
      LogTrace(metname) << "     MuRingForwardLayer::fastCompatibleDets:PreviousRing:" << idet
			<< " R1 " << theRings[idet]->specificSurface().innerRadius()
			<< " R2: " << theRings[idet]->specificSurface().outerRadius()
			<< " FTS R " << nextPos.perp();
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
   
  LogTrace(metname) << "     MuRingForwardLayer::fastCompatibleDets: found: "
		    << result.size()
		    << " on closest: " << nclosest
		    << " # checked rings: " << 1 + nnextdet;
   
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


GeomDetEnumerators::SubDetector MuRingForwardLayer::subDetector() const {
  return theBasicComps.front()->subDetector();
}

const vector<const GeometricSearchDet*> &
MuRingForwardLayer::components() const {
  return theComponents;
}
