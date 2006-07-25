/** \file
 *
 *  $Date: 2006/06/13 08:46:03 $
 *  $Revision: 1.7 $
 *  \author N. Amapane - CERN
 */


#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "Utilities/BinningTools/interface/PeriodicBinFinderInPhi.h"

#include "GeneralBinFinderInPhi.h"
#include "PhiBorderFinder.h"

#include <algorithm>
#include <iostream>

#define MDEBUG false //FIXME!

using namespace std;

MuRodBarrelLayer::MuRodBarrelLayer(vector<const DetRod*>& rods) :
  theRods(rods),
  theComponents(theRods.begin(),theRods.end()),
  theBinFinder(0),
  isOverlapping(false)
{
  // Cache chamber pointers (the basic components_)
  for (vector<const DetRod*>::const_iterator it=rods.begin();
       it!=rods.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(),tmp2.begin(),tmp2.end());
  }

  // Initialize the binfinder
  PhiBorderFinder bf(theRods);
  isOverlapping = bf.isPhiOverlapping();

  if ( bf.isPhiPeriodic() ) { 
    theBinFinder = new PeriodicBinFinderInPhi<double>
    (theRods.front()->position().phi(),theRods.size());
  } else {
    theBinFinder = new GeneralBinFinderInPhi<double>(bf);
  }

  // Compute the layer's surface and bounds (from the components())
  BarrelDetLayer::initialize(); 

  if ( MDEBUG ) 
    cout << "Constructing MuRodBarrelLayer: "
	 << basicComponents().size() << " Dets " 
	 << theRods.size() << " Rods "
	 << " R: " << specificSurface().radius()
	 << " Per.: " << bf.isPhiPeriodic()
	 << " Overl.: " << isOverlapping
	 << endl;
}


MuRodBarrelLayer::~MuRodBarrelLayer() {
  delete theBinFinder;
  for (vector <const DetRod*>::iterator i = theRods.begin();
       i<theRods.end(); i++) {delete *i;}
}


vector<GeometricSearchDet::DetWithState> 
MuRodBarrelLayer::compatibleDets(const TrajectoryStateOnSurface& startingState,
				 const Propagator& prop, 
				 const MeasurementEstimator& est) const {
  vector<DetWithState> result; 

  if ( MDEBUG ) 
    cout << "MuRodBarrelLayer::compatibleDets, Cyl R: " 
	 << specificSurface().radius()
	 << " TSOS at R: " << startingState.globalPosition().perp()
	 << endl;

  pair<bool, TrajectoryStateOnSurface> compat =
    compatible(startingState, prop, est);
  if (!compat.first) {
    if ( MDEBUG )
      cout << "     MuRodBarrelLayer::compatibleDets: not compatible"
	   << " (should not have been selected!)" <<endl;
    return vector<DetWithState>();
  } 


  TrajectoryStateOnSurface& tsos = compat.second;

  int closest = theBinFinder->binIndex(tsos.globalPosition().phi());
  const DetRod* closestRod = theRods[closest];

  // Check the closest rod
  if ( MDEBUG ) 
    cout << "     MuRodBarrelLayer::compatibleDets, closestRod: " << closest
	 << " phi : " << closestRod->surface().position().phi()
	 << " FTS phi: " << tsos.globalPosition().phi()
	 << endl;

  result = closestRod->compatibleDets(tsos, prop, est);

  int nclosest = result.size(); // Debug counter

  bool checknext = false ;
  double dist;

  if (!result.empty()) { 
    // Check if the track go outside closest rod, then look for closest. 
    TrajectoryStateOnSurface& predictedState = result.front().second;
    float xErr = xError(predictedState, est);
    float halfWid = closestRod->surface().bounds().width()/2.;
    dist = predictedState.localPosition().x();

    // If the layer is overlapping, additionally reduce halfWid by 10%
    // to account for overlap.
    // FIXME: should we account for the real amount of overlap?
    if (isOverlapping) halfWid *= 0.9;

    if (fabs(dist) + xErr > halfWid) {
      checknext = true;
    }
  } else { // Rod is not compatible
    //FIXME: Usually next cannot be either. Implement proper logic.
    // (in general at least one rod should be when this method is called by
    // compatibleDets() which calls compatible())
    checknext = true;
    
    // Look for the next-to closest in phi.
    // Note Geom::Phi, subtraction is pi-border-safe
    if ( tsos.globalPosition().phi()-closestRod->surface().position().phi()>0.)
    {
      dist = -1.;
    } else {
      dist = +1.;
    }

    if ( MDEBUG ) 
      cout << "     MuRodBarrelLayer::fastCompatibleDets, none on closest rod!"
	   << endl;
  }

  if (checknext) {
    int next;
    if (dist<0.) next = closest+1;
    else next = closest-1;

    next = theBinFinder->binIndex(next); // Bin Periodicity
    const DetRod* nextRod = theRods[next];

    if ( MDEBUG ) 
      cout << "     MuRodBarrelLayer::fastCompatibleDets, next-to closest"
	   << " rod: " << next << " dist " << dist
	   << " phi : " << nextRod->surface().position().phi()
	   << " FTS phi: " << tsos.globalPosition().phi()
	   << endl;    
    
    vector<DetWithState> nextRodDets =
      nextRod->compatibleDets(tsos, prop, est);
    result.insert(result.end(), 
		  nextRodDets.begin(), nextRodDets.end());
  }
  
  if ( MDEBUG ) 
    cout << "     MuRodBarrelLayer::fastCompatibleDets: found: "
	 << result.size()
	 << " on closest: " << nclosest
	 << " # checked rods: " << 1 + int(checknext)
	 << endl;
  
  return result;
}


vector<DetGroup> 
MuRodBarrelLayer::groupedCompatibleDets( const TrajectoryStateOnSurface& startingState,
					 const Propagator& prop,
					 const MeasurementEstimator& est) const {
  // FIXME should return only 1 group 
  cout << "dummy implementation of MuRodBarrelLayer::groupedCompatibleDets()" << endl;
  return vector<DetGroup>();
}


bool MuRodBarrelLayer::hasGroups() const {
  // FIXME : depending on isOverlapping?
  return false;
}


GeomDetEnumerators::SubDetector MuRodBarrelLayer::subDetector() const {
  return theBasicComps.front()->subDetector();
}

const vector<const GeometricSearchDet*>&
MuRodBarrelLayer::components() const {
  return theComponents;
}

float MuRodBarrelLayer::xError(const TrajectoryStateOnSurface& tsos,
			       const MeasurementEstimator& est) const {
  const float nSigmas = 3.f;
  if (tsos.hasError()) {
    return nSigmas * sqrt(tsos.localError().positionError().xx());
  }
  else return nSigmas * 0.5;
}
