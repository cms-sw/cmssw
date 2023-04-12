#ifndef DATAFORMATS_TRACKERRECHIT2D_PHASE2TRACKERRECHIT1D_H
#define DATAFORMATS_TRACKERRECHIT2D_PHASE2TRACKERRECHIT1D_H

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

#include "TkCloner.h"

class Phase2TrackerRecHit1D final : public TrackerSingleRecHit {
public:
  typedef OmniClusterRef::Phase2Cluster1DRef ClusterRef;

  Phase2TrackerRecHit1D() {}

  ~Phase2TrackerRecHit1D() override {}

  Phase2TrackerRecHit1D(const LocalPoint& pos, const LocalError& err, GeomDet const& idet, ClusterRef const& clus)
      : TrackerSingleRecHit(pos, err, idet, clus) {}

  Phase2TrackerRecHit1D* clone() const override { return new Phase2TrackerRecHit1D(*this); }
  RecHitPointer cloneSH() const override { return std::make_shared<Phase2TrackerRecHit1D>(*this); }

  ClusterRef cluster() const { return cluster_phase2OT(); }
  void setClusterRef(ClusterRef const& ref) { setClusterPhase2Ref(ref); }

  bool isPhase2() const override { return true; }
  //FIXME::check dimension of this!!
  int dimension() const override { return 2; }
  void getKfComponents(KfComponentsHolder& holder) const override { getKfComponents2D(holder); }

  bool canImproveWithTrack() const override { return true; }

private:
  // double dispatch
  Phase2TrackerRecHit1D* clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this, tsos).release();
  }
  RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this, tsos);
  }
};

typedef edmNew::DetSetVector<Phase2TrackerRecHit1D> Phase2TrackerRecHit1DCollectionNew;

#endif
