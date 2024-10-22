#ifndef TrackerSingleRecHit_H
#define TrackerSingleRecHit_H

#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

/*  a Hit composed by a "single" measurement
 * it has a reference to a cluster and a local position&error
 */
class TrackerSingleRecHit : public BaseTrackerRecHit {
public:
  typedef BaseTrackerRecHit Base;

  TrackerSingleRecHit() {}

  typedef OmniClusterRef::ClusterPixelRef ClusterPixelRef;
  typedef OmniClusterRef::ClusterStripRef ClusterStripRef;
  typedef OmniClusterRef::Phase2Cluster1DRef ClusterPhase2Ref;
  typedef OmniClusterRef::ClusterMTDRef ClusterMTDRef;

  // no position (as in persistent)
  TrackerSingleRecHit(DetId id, OmniClusterRef const& clus) : Base(id, trackerHitRTTI::single), cluster_(clus) {}

  template <typename CluRef>
  TrackerSingleRecHit(const LocalPoint& p, const LocalError& e, GeomDet const& idet, CluRef const& clus)
      : Base(p, e, idet, trackerHitRTTI::single), cluster_(clus) {}

  // for projected or timing...
  template <typename CluRef>
  TrackerSingleRecHit(
      const LocalPoint& p, const LocalError& e, GeomDet const& idet, trackerHitRTTI::RTTI rt, CluRef const& clus)
      : Base(p, e, idet, rt), cluster_(clus) {}

  // a single hit is on a detunit
  const GeomDetUnit* detUnit() const override { return det(); }

  // used by trackMerger (to be improved)
  OmniClusterRef const& firstClusterRef() const final { return cluster_; }

  OmniClusterRef const& omniClusterRef() const { return cluster_; }
  OmniClusterRef const& omniCluster() const { return cluster_; }
  // for rekeying...
  OmniClusterRef& omniClusterRef() { return cluster_; }
  OmniClusterRef& omniCluster() { return cluster_; }

  ClusterPixelRef cluster_pixel() const { return cluster_.cluster_pixel(); }

  ClusterStripRef cluster_strip() const { return cluster_.cluster_strip(); }

  ClusterPhase2Ref cluster_phase2OT() const { return cluster_.cluster_phase2OT(); }

  ClusterMTDRef cluster_mtd() const { return cluster_.cluster_mtd(); }

  SiStripCluster const& stripCluster() const { return cluster_.stripCluster(); }

  SiPixelCluster const& pixelCluster() const { return cluster_.pixelCluster(); }

  Phase2TrackerCluster1D const& phase2OTCluster() const { return cluster_.phase2OTCluster(); }

  FTLCluster const& mtdCluster() const { return cluster_.mtdCluster(); }

  // void setClusterRef(const &  OmniClusterRef ref) {  cluster_ =ref;}
  void setClusterPixelRef(ClusterPixelRef const& ref) { cluster_ = OmniClusterRef(ref); }
  void setClusterStripRef(ClusterStripRef const& ref) { cluster_ = OmniClusterRef(ref); }
  void setClusterPhase2Ref(ClusterPhase2Ref const& ref) { cluster_ = OmniClusterRef(ref); }
  void setClusterMTDRef(ClusterMTDRef const& ref) { cluster_ = OmniClusterRef(ref); }

  bool sharesInput(const TrackingRecHit* other, SharedInputType what) const final;

  bool sharesInput(TrackerSingleRecHit const& other) const { return cluster_ == other.cluster_; }

  bool sameCluster(OmniClusterRef const& oh) const { return oh == cluster_; }

  std::vector<const TrackingRecHit*> recHits() const override;
  std::vector<TrackingRecHit*> recHits() override;

private:
  // new game
  OmniClusterRef cluster_;
};

#endif
