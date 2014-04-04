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

#ifdef NO_DICT
  TrackingRecHit::ConstRecHitPointer makeShared(TrackingRecHit::ConstRecHitPointer const & hit, TrajectoryStateOnSurface const& tsos) const {
    return hit->canImproveWithTrack() ?  hit->cloneSH(*this, tsos) : hit;
    // return  hit->cloneSH(*this, tsos);
  }
#endif

  virtual SiPixelRecHit * operator()(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual SiStripRecHit2D * operator()(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual SiStripRecHit1D * operator()(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual SiStripMatchedRecHit2D * operator()(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual ProjectedSiStripRecHit2D * operator()(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;

#ifdef NO_DICT
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
#endif

};
#endif
