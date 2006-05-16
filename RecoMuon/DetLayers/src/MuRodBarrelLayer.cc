/** \file
 *
 *  $Date: 2006/04/25 17:03:23 $
 *  $Revision: 1.2 $
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
  //  RodBarrelLayer(rods), FIXME: should be removed?
  theRods(rods),
  isOverlapping(false) 
{
  // Cache chamber pointers (the basic components_)
  for (vector<const DetRod*>::const_iterator it=rods.begin();
       it!=rods.end(); it++) {
    vector<const GeomDet*> tmp2 = (*it)->basicComponents();
    theBasicComps.insert(theBasicComps.end(),tmp2.begin(),tmp2.end());
  }

  // Initialize the binfinder
  PhiBorderFinder bf(basicComponents());
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


Module MuRodBarrelLayer::module() const {
  return dt;
}

vector<const GeometricSearchDet*> 
MuRodBarrelLayer::components() const {
  return vector <const GeometricSearchDet*>(theRods.begin(),theRods.end());
}
