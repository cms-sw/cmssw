#include "RecoMuon/TrackerSeedGenerator/interface/PrimitiveMuonSeed.h"

/**  \class PrimitiveMuonSeed
 *
 *   Primitive tracker seed from muon which
 *   consists of one measurement and a trajectory state
 *
 *   $Date: 2006/07/26 18:24:02 $
 *   $Revision: 1.3 $
 *
 *   \author   N. Neumeister   - Purdue University
 */

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

