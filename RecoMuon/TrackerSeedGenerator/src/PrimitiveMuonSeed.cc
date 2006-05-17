#include "RecoMuon/TrackerSeedGenerator/interface/PrimitiveMuonSeed.h"

/**  \class PrimitiveMuonSeed
 *
 *   Primitive tracker seed from muon which
 *   consists of one measurement and a trajectory state
 *
 *   $Date: $
 *   $Revision: $
 *
 *   \author   N. Neumeister   - Purdue University
 */

//
//
PrimitiveMuonSeed::PrimitiveMuonSeed(const PTrajectoryStateOnDet& state,
                                     const PropagationDirection direction,
//                                     const DetLayer* layer,
                                     const TrajectoryMeasurement& meas) :
     theState(state), theDirection(direction), theMeasurement(meas) {
  
}

PTrajectoryStateOnDet PrimitiveMuonSeed::startingState() const {

  return theState;
}

//
//
PropagationDirection PrimitiveMuonSeed::direction() const {

  return theDirection;

}

  BasicTrajectorySeed::range PrimitiveMuonSeed::recHits() const {

  range result;
  //FIXME
  return result;

}


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
bool PrimitiveMuonSeed::share(const BasicTrajectorySeed&) const {

  return false;
  
}

//
//
BasicTrajectorySeed* PrimitiveMuonSeed::clone() const {

  return new PrimitiveMuonSeed(*this);
  
}

