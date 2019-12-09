#ifndef ProjectedSiStripRecHit2D_H
#define ProjectedSiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

// #include<iostream>

class ProjectedSiStripRecHit2D final : public TrackerSingleRecHit {
public:
  inline static bool isMono(GeomDet const& gdet, GeomDet const& sdet) {
    return (sdet.geographicalId() - gdet.geographicalId()) == 2;
  }

  typedef TrackerSingleRecHit Base;

  ProjectedSiStripRecHit2D() : theOriginalDet(nullptr) {}

  ProjectedSiStripRecHit2D(const LocalPoint& pos,
                           const LocalError& err,
                           GeomDet const& idet,
                           SiStripRecHit2D const& originalHit)
      : TrackerSingleRecHit(pos,
                            err,
                            idet,
                            isMono(idet, *originalHit.det()) ? trackerHitRTTI::projMono : trackerHitRTTI::projStereo,
                            originalHit.omniCluster()),
        theOriginalDet(originalHit.det()) {
    //      std::cout << getRTTI() << ' ' << originalHit.rawId() << ' ' << idet.geographicalId().rawId() << ' ' << originalId() << std::endl;
    assert(originalId() == originalHit.rawId());
  }

  template <typename CluRef>
  ProjectedSiStripRecHit2D(
      const LocalPoint& pos, const LocalError& err, GeomDet const& idet, GeomDet const& originalDet, CluRef const& clus)
      : TrackerSingleRecHit(
            pos, err, idet, isMono(idet, originalDet) ? trackerHitRTTI::projMono : trackerHitRTTI::projStereo, clus),
        theOriginalDet(&originalDet) {
    assert(originalId() == originalDet.geographicalId());
  }

  void setDet(const GeomDet& idet) override;

  bool canImproveWithTrack() const override { return true; }

  ProjectedSiStripRecHit2D* clone() const override { return new ProjectedSiStripRecHit2D(*this); }

  int dimension() const override { return 2; }
  void getKfComponents(KfComponentsHolder& holder) const override { getKfComponents2D(holder); }

  typedef OmniClusterRef::ClusterStripRef ClusterRef;
  ClusterRef cluster() const { return cluster_strip(); }
  const GeomDetUnit* originalDet() const { return theOriginalDet; }
  unsigned int originalId() const { return trackerHitRTTI::projId(*this); }

  // not useful only for backward compatibility
  SiStripRecHit2D originalHit() const { return SiStripRecHit2D(originalId(), omniClusterRef()); }

  std::vector<const TrackingRecHit*> recHits() const override {
    std::vector<const TrackingRecHit*> rechits;
    return rechits;
  }
  std::vector<TrackingRecHit*> recHits() override {
    std::vector<TrackingRecHit*> rechits;
    return rechits;
  }

private:
  // double dispatch
  ProjectedSiStripRecHit2D* clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this, tsos).release();
  }
#ifndef __GCCXML__
  ConstRecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this, tsos);
  }
#endif

private:
  const GeomDet* theOriginalDet;
};

#endif
