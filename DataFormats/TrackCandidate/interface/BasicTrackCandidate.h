#ifndef DATAFORMATS_TRACKCANDIDATE_BASICTRACKCANDIDATE_H
#define DATAFORMATS_TRACKCANDIDATE_BASICTRACKCANDIDATE_H

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h.h"

/** A track candidate is
    - a TSOS or equivalent (here a PTrajectoryStateOnDet)
    - a vector of rechits (here via the OwnVector interface)
    - a TrajectorySeed (to be confirmed as matching the final track)

only the second is compulsory,the other two can be empty / not present
**/

class BasicTrackCandidate{
 public:
  typedef OwnVector<TrackingRecHit> RecHitContainer;
  typedef recHitContainer::iterator iterator;
  typedef std::pair<iterator,iterator> range;
  
  BasicTrackCandidate(){}

  virtual PTrajectoryStateOnDet& trajectoryStateOnDet() const = 0;
  virtual range recHits() const = 0;
  virtual TrajectorySeed& seed() const = 0;
};
