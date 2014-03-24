#ifndef DATAFORMATS_TRACKCANDIDATE_TRACKCANDIDATE_H
#define DATAFORMATS_TRACKCANDIDATE_TRACKCANDIDATE_H

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"


#include <utility>

/** A track candidate is
    - a TSOS or equivalent (here a PTrajectoryStateOnDet)
    - a vector of rechits (here via the OwnVector interface)
    - a TrajectorySeed (to be confirmed as matching the final track)
    - a reference to the TrajectorySeed in the origianl collection 
      of seeds. Often this collection is not saved on disk and 
      therefore the reference may be invalid.

only the second is compulsory,the other three can be empty / not present
**/

class TrackCandidate{
public:
  typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
  typedef RecHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;
  
 TrackCandidate(): rh_(),  seed_(), state_(), seedRef_(), nLoops_(0) {}

  explicit TrackCandidate(RecHitContainer & rh) :
  rh_(),  seed_(), state_(), seedRef_(), nLoops_(0) {rh_.swap(rh);}
  
  TrackCandidate(RecHitContainer & rh,
		 TrajectorySeed const & s,
		 PTrajectoryStateOnDet const & st,
		 signed char nLoops=0):
    rh_(), seed_(s), state_(st), seedRef_(),nLoops_(nLoops) {rh_.swap(rh);}

  
  TrackCandidate(RecHitContainer & rh,
		 TrajectorySeed const & s,
		 PTrajectoryStateOnDet const & st,
		 const edm::RefToBase<TrajectorySeed> & seedRef,
		 signed char nLoops=0) :
    rh_(), seed_(s), state_(st), seedRef_(seedRef),nLoops_(nLoops) {rh_.swap(rh);}


  PTrajectoryStateOnDet const & trajectoryStateOnDet() const { return state_;}
  
  range recHits() const {return std::make_pair(rh_.begin(), rh_.end());}
  
  TrajectorySeed const & seed() const {return seed_;}
 

  bool isLooper() const {return (nLoops_>0);}
  signed char nLoops() const {return nLoops_;}

  void setNLoops(signed char value) {nLoops_=value;}

  /**  return the edm::reference to the trajectory seed in the original
   *   seeds collection. If the collection has been dropped from the
   *   Event, the reference may be invalid. Its validity should be tested,
   *   before the reference is actually used. 
   */
  edm::RefToBase<TrajectorySeed> seedRef() const { return seedRef_; }

  void setSeedRef(edm::RefToBase<TrajectorySeed> & seedRef) { seedRef_ = seedRef ; } 
  
private:
  RecHitContainer rh_;
  TrajectorySeed seed_;
  PTrajectoryStateOnDet state_;
  edm::RefToBase<TrajectorySeed> seedRef_;
  signed char nLoops_;
};
#endif
