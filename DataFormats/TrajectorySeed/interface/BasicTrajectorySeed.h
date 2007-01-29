#ifndef DATAFORMATS_TRAJECTORYSEED_BASICTRAJECTORYSEED_h
#define DATAFORMATS_TRAJECTORYSEED_BASICTRAJECTORYSEED_h

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"

/**
   BasicTrajectorySeed contains
   - a TSOS (a PersistentTrajectoryStateonDet)
   - a vector of RecHits (with Own_vector to store polimorphic) via a range
   - a propagation direction
**/
class BasicTrajectorySeed {
public:
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef recHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;
  
  BasicTrajectorySeed() { }

  virtual BasicTrajectorySeed* clone () const = 0;
  
  // returns the recHits
  
  virtual range recHits() const = 0;
  virtual PropagationDirection direction() const = 0;
  virtual PTrajectoryStateOnDet startingState() const = 0;
};

#endif
