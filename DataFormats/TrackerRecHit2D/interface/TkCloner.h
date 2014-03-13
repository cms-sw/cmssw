#ifndef TKClonerRecHit_H
#define TKClonerRecHit_H

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

class SiPixelRecHit;
class SiStripRecHit2D;
class SiStripRecHit1D;
class SiStripMatchedRecHit2D;
class ProjectedSiStripRecHit2D;

class TkCloner {
public:
  TrackingRecHit * operator()(TrackingRecHit const & hit, TrajectoryStateOnSurface const& tsos) const {
    return hit.clone(*this, tsos);
  }

  virtual SiPixelRecHit * operator()(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual SiStripRecHit2D * operator()(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual SiStripRecHit1D * operator()(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual SiStripMatchedRecHit2D * operator()(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual ProjectedSiStripRecHit2D * operator()(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;

};
#endif
