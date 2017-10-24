#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

// #include<iostream>

void ProjectedSiStripRecHit2D::setDet(std::shared_ptr<const GeomDet> idet) {
    TrackingRecHit::setDet(idet);
    std::shared_ptr<const GluedGeomDet> gdet = std::static_pointer_cast<const GluedGeomDet>(idet);
    theOriginalDet = trackerHitRTTI::isProjMono(*this) ? gdet->monoDet() : gdet->stereoDet();
}

