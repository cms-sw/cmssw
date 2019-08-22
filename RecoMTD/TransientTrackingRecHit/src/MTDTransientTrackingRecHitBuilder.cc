/**
 *  Class: MTDTransientTrackingRecHitBuilder
 *
 *  Description:
 *
 *
 *
 *  Authors :
 *  L. Gray               FNAL
 *
 **/

#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

MTDTransientTrackingRecHitBuilder::MTDTransientTrackingRecHitBuilder(
    edm::ESHandle<GlobalTrackingGeometry> trackingGeometry)
    : theTrackingGeometry(trackingGeometry) {}

MTDTransientTrackingRecHitBuilder::RecHitPointer MTDTransientTrackingRecHitBuilder::build(
    const TrackingRecHit* p, edm::ESHandle<GlobalTrackingGeometry> trackingGeometry) const {
  if (p->geographicalId().det() == DetId::Forward && p->geographicalId().subdetId() == FastTime) {
    return p->cloneSH();
  }

  return RecHitPointer();
}

MTDTransientTrackingRecHitBuilder::RecHitPointer MTDTransientTrackingRecHitBuilder::build(
    const TrackingRecHit* p) const {
  if (theTrackingGeometry.isValid())
    return build(p, theTrackingGeometry);
  else
    throw cms::Exception("MTD|RecoMTD|MTDTransientTrackingRecHitBuilder")
        << "ERROR! You are trying to build a MTDTransientTrackingRecHit with a non valid GlobalTrackingGeometry";
}

MTDTransientTrackingRecHitBuilder::ConstRecHitContainer MTDTransientTrackingRecHitBuilder::build(
    const trackingRecHit_iterator& start, const trackingRecHit_iterator& stop) const {
  ConstRecHitContainer result;
  for (trackingRecHit_iterator hit = start; hit != stop; ++hit)
    result.push_back(build(&**hit));

  return result;
}
