#ifndef DATAFORMATS_TRAJECTORYSEED_TRAJECTORYSEED_h
#define DATAFORMATS_TRAJECTORYSEED_TRAJECTORYSEED_h

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "FWCore/Utilities/interface/Range.h"
#include <utility>
#include <algorithm>

/**
   TrajectorySeed contains
   - a TSOS
   - a vector of RecHits (with Own_vector to store polimorphic)
   - a propagation direction
**/
class TrajectorySeed {
public:
  typedef edm::OwnVector<TrackingRecHit> RecHitContainer;
  typedef edm::Range<RecHitContainer::const_iterator> RecHitRange;

  TrajectorySeed() {}
  virtual ~TrajectorySeed() {}

  TrajectorySeed(PTrajectoryStateOnDet const& ptsos, RecHitContainer const& rh, PropagationDirection dir)
      : hits_(rh), tsos_(ptsos), dir_(dir) {}

  TrajectorySeed(PTrajectoryStateOnDet const& ptsos, RecHitContainer&& rh, PropagationDirection dir) noexcept
      : hits_(std::move(rh)), tsos_(ptsos), dir_(dir) {}

  void swap(PTrajectoryStateOnDet& ptsos, RecHitContainer& rh, PropagationDirection& dir) noexcept {
    hits_.swap(rh);
    std::swap(tsos_, ptsos);
    std::swap(dir_, dir);
  }

  void swap(TrajectorySeed& rh) noexcept {
    hits_.swap(rh.hits_);
    std::swap(tsos_, rh.tsos_);
    std::swap(dir_, rh.dir_);
  }

  TrajectorySeed(TrajectorySeed const& o) = default;

  TrajectorySeed& operator=(TrajectorySeed const& o) = default;

  TrajectorySeed(TrajectorySeed&& o) noexcept = default;

  TrajectorySeed& operator=(TrajectorySeed&& o) noexcept = default;

  RecHitRange recHits() const { return {hits_.begin(), hits_.end()}; }
  unsigned int nHits() const { return hits_.size(); }
  PropagationDirection direction() const { return dir_; }
  PTrajectoryStateOnDet const& startingState() const { return tsos_; }

  virtual TrajectorySeed* clone() const { return new TrajectorySeed(*this); }

private:
  RecHitContainer hits_;
  PTrajectoryStateOnDet tsos_;
  PropagationDirection dir_;
};

inline void swap(TrajectorySeed& rh, TrajectorySeed& lh) noexcept { rh.swap(lh); }

typedef TrajectorySeed BasicTrajectorySeed;

#endif
