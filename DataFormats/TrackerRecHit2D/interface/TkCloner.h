#ifndef TKClonerRecHit_H
#define TKClonerRecHit_H

#include <memory>
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Utilities/interface/HideStdUniquePtrFromRoot.h"

class SiPixelRecHit;
class SiStripRecHit2D;
class SiStripRecHit1D;
class SiStripMatchedRecHit2D;
class ProjectedSiStripRecHit2D;

class TkCloner {
public:
  TrackingRecHit * operator() CMS_THREAD_SAFE (TrackingRecHit const & hit, TrajectoryStateOnSurface const& tsos) const {
    return hit.clone(*this, tsos);
  }

#ifndef __GCCXML__
  TrackingRecHit::ConstRecHitPointer makeShared(TrackingRecHit::ConstRecHitPointer const & hit, TrajectoryStateOnSurface const& tsos) const {
    return hit->canImproveWithTrack() ?  hit->cloneSH(*this, tsos) : hit;
    // return  hit->cloneSH(*this, tsos);
  }
#endif

  virtual std::unique_ptr<SiPixelRecHit> operator()(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual std::unique_ptr<SiStripRecHit2D> operator()(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual std::unique_ptr<SiStripRecHit1D> operator()(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual std::unique_ptr<SiStripMatchedRecHit2D> operator()(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual std::unique_ptr<ProjectedSiStripRecHit2D> operator()(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;

#ifndef __GCCXML__
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiPixelRecHit const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripRecHit1D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(SiStripMatchedRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
  virtual TrackingRecHit::ConstRecHitPointer makeShared(ProjectedSiStripRecHit2D const & hit, TrajectoryStateOnSurface const& tsos) const=0;
#endif

};
#endif
