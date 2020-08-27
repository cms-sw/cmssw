/** \file
 *
 */

#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHit.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/AlignmentPositionError.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <map>

typedef MTDTransientTrackingRecHit::MTDRecHitPointer MTDRecHitPointer;
typedef MTDTransientTrackingRecHit::RecHitContainer MTDRecHitContainer;

MTDTransientTrackingRecHit::MTDTransientTrackingRecHit(const GeomDet* geom, const TrackingRecHit* rh)
    : GenericTransientTrackingRecHit(*geom, *rh) {}

MTDTransientTrackingRecHit::MTDTransientTrackingRecHit(const MTDTransientTrackingRecHit& other)
    : GenericTransientTrackingRecHit(*other.det(), *(other.hit())) {}

bool MTDTransientTrackingRecHit::isBTL() const {
  MTDDetId temp(geographicalId());
  return (temp.mtdSubDetector() == MTDDetId::BTL);
}

bool MTDTransientTrackingRecHit::isETL() const {
  MTDDetId temp(geographicalId());
  return (temp.mtdSubDetector() == MTDDetId::ETL);
}

void MTDTransientTrackingRecHit::invalidateHit() {
  setType(bad);  //trackingRecHit_->setType(bad); // maybe add in later
}
