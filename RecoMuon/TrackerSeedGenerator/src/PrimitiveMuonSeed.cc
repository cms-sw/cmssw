/**  \class PrimitiveMuonSeed
 *
 *   Primitive tracker seed from muon which
 *   consists of one measurement and a trajectory state
 *
 *   $Date: 2006/07/26 20:26:47 $
 *   $Revision: 1.4 $
 *
 *   \author   N. Neumeister   - Purdue University
 */

#include "RecoMuon/TrackerSeedGenerator/interface/PrimitiveMuonSeed.h"
#include <utility>

//
//
//
PrimitiveMuonSeed::PrimitiveMuonSeed(const PTrajectoryStateOnDet& state,
                                     const PropagationDirection& direction,
				     const recHitContainer layerRecHits,
				     const TrajectoryMeasurement& meas) :
  theTrajectorySeed(state,layerRecHits,direction),
  theMeasurement(meas) 
{
  
}


//
//
//
std::vector<TrajectoryMeasurement> PrimitiveMuonSeed::measurements() const {

  std::vector<TrajectoryMeasurement> result;
  result.reserve(1);

  result.push_back(theMeasurement);

  return result;
  
}


//
//
//
bool PrimitiveMuonSeed::share(const BasicTrajectorySeed&) const {

  return false;
  
}


//
//
//
PropagationDirection PrimitiveMuonSeed::direction() const {

  return theDirection;

}


//
//
//
PrimitiveMuonSeed* PrimitiveMuonSeed::clone() const {

  return new PrimitiveMuonSeed(*this);
  
}


//
//
//
PTrajectoryStateOnDet PrimitiveMuonSeed::startingState() const {

  return theState;

}


//
//
//
BasicTrajectorySeed::range PrimitiveMuonSeed::recHits() const {

  range result;
  return result;

}

