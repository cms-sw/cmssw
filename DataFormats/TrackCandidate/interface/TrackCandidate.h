#ifndef DATAFORMATS_TRACKCANDIDATE_TRACKCANDIDATE_H
#define DATAFORMATS_TRACKCANDIDATE_TRACKCANDIDATE_H
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackCandidate/interface/BasicTrackCandidate.h"


#include <utility>

/** A track candidate is
    - a TSOS or equivalent (here a PTrajectoryStateOnDet)
    - a vector of rechits (here via the OwnVector interface)
    - a TrajectorySeed (to be confirmed as matching the final track)

only the second is compulsory,the other two can be empty / not present
**/

class TrackCandidate{
 public:
  typedef BasicTrackCandidate::RecHitContainer RecHitContainer;
  typedef BasicTrackCandidate::iterator iterator;
  typedef BasicTrackCandidate::range range;
  
  TrackCandidate(){}
 virtual ~TrackCandidate(){}
  
  
  TrackCandidate(RecHitContainer rh, TrajectorySeed* s, PTrajectoryStateOnDet* st) :
    rh_(rh), seed_(s), state_(st) {}
  TrackCandidate(RecHitContainer rh) :
    rh_(rh),  seed_(0), state_(0) {}
  
  PTrajectoryStateOnDet& trajectoryStateOnDet() const { return *state_;}

  range recHits() {return std::make_pair(rh_.begin(), rh_.end());}

  TrajectorySeed& seed() const {return *seed_;}

 private:
  RecHitContainer rh_;
  TrajectorySeed* seed_;
  PTrajectoryStateOnDet* state_;
};
#endif
