#include "Geometry/CommonTopologies/interface/GeomDetType.h"

using namespace GeomDetEnumerators;

GeomDetType::GeomDetType(const std::string& n, SubDetector subdet) : theName(n), theSubDet(subdet) {}

GeomDetType::~GeomDetType() {}

bool GeomDetType::isBarrel() const { return GeomDetEnumerators::isBarrel(theSubDet); }

bool GeomDetType::isEndcap() const { return GeomDetEnumerators::isEndcap(theSubDet); }

bool GeomDetType::isTrackerStrip() const { return GeomDetEnumerators::isTrackerStrip(theSubDet); }

bool GeomDetType::isTrackerPixel() const { return GeomDetEnumerators::isTrackerPixel(theSubDet); }

bool GeomDetType::isInnerTracker() const { return GeomDetEnumerators::isInnerTracker(theSubDet); }

bool GeomDetType::isOuterTracker() const { return GeomDetEnumerators::isOuterTracker(theSubDet); }

bool GeomDetType::isTracker() const { return GeomDetEnumerators::isTracker(theSubDet); }

bool GeomDetType::isDT() const { return GeomDetEnumerators::isDT(theSubDet); }

bool GeomDetType::isCSC() const { return GeomDetEnumerators::isCSC(theSubDet); }

bool GeomDetType::isRPC() const { return GeomDetEnumerators::isRPC(theSubDet); }

bool GeomDetType::isGEM() const { return GeomDetEnumerators::isGEM(theSubDet); }

bool GeomDetType::isME0() const {
  return GeomDetEnumerators::isME0(theSubDet);
  return (theSubDet == ME0);
}

bool GeomDetType::isMuon() const { return GeomDetEnumerators::isMuon(theSubDet); }

bool GeomDetType::isTiming() const { return GeomDetEnumerators::isTiming(theSubDet); }
