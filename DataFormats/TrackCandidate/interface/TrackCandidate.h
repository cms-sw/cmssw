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
  typedef BasicTrackCandidate::const_iterator const_iterator;
  typedef BasicTrackCandidate::range range;
  
  TrackCandidate(){}
  virtual ~TrackCandidate(){}
  
  
  TrackCandidate(RecHitContainer & rh, TrajectorySeed const & s, PTrajectoryStateOnDet const & st) :
    rh_(), seed_(s), state_(st) {rh_.swap(rh);}
  TrackCandidate(RecHitContainer & rh) :
    rh_(),  seed_(), state_() {rh_.swap(rh);}
  
  PTrajectoryStateOnDet const & trajectoryStateOnDet() const { return state_;}
  
  range recHits() const {return std::make_pair(rh_.begin(), rh_.end());}
  
  TrajectorySeed const & seed() const {return seed_;}
  
private:
  RecHitContainer rh_;
  TrajectorySeed seed_;
  PTrajectoryStateOnDet state_;
};
#endif
