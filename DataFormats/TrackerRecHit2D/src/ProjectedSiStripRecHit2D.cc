#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

// #include<iostream>

void ProjectedSiStripRecHit2D::setDet(const GeomDet & idet) {
    TrackingRecHit::setDet(idet);
    const GluedGeomDet& gdet = static_cast<const GluedGeomDet&>(idet);
    theOriginalDet = trackerHitRTTI::isProjMono(*this) ? gdet.monoDet() : gdet.stereoDet();
}

