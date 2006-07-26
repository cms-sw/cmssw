#include "RecoMuon/TrackerSeedGenerator/interface/PrimitiveMuonSeed.h"

/**  \class PrimitiveMuonSeed
 *
 *   Primitive tracker seed from muon which
 *   consists of one measurement and a trajectory state
 *
 *   $Date: 2006/07/10 13:20:35 $
 *   $Revision: 1.2 $
 *
 *   \author   N. Neumeister   - Purdue University
 */

//
//
PrimitiveMuonSeed::PrimitiveMuonSeed(const PTrajectoryStateOnDet& state,
                                     const PropagationDirection& direction,
				     const recHitContainer layerRecHits,
				     const TrajectoryMeasurement& meas) :
  TrajectorySeed(state,layerRecHits,direction),
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
PrimitiveMuonSeed* PrimitiveMuonSeed::clone() const {

  return new PrimitiveMuonSeed(*this);
  
}

