#ifndef RecoMTD_MTDTransientTrackingRecHit_MTDTransientTrackingRecHitBuilder_h
#define RecoMTD_MTDTransientTrackingRecHit_MTDTransientTrackingRecHitBuilder_h

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "FWCore/Framework/interface/ESHandle.h"

class MTDTransientTrackingRecHitBuilder : public TransientTrackingRecHitBuilder {
public:
  typedef TransientTrackingRecHit::RecHitPointer RecHitPointer;
  typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

  MTDTransientTrackingRecHitBuilder(edm::ESHandle<GlobalTrackingGeometry> trackingGeometry = nullptr);

  ~MTDTransientTrackingRecHitBuilder() override{};

  using TransientTrackingRecHitBuilder::build;
  /// Call the MTDTransientTrackingRecHit::specificBuild
  RecHitPointer build(const TrackingRecHit* p, edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const;

  RecHitPointer build(const TrackingRecHit* p) const override;

  ConstRecHitContainer build(const trackingRecHit_iterator& start, const trackingRecHit_iterator& stop) const;

private:
  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
};

#endif
