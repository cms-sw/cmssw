#include "Geometry/TrackerNumberingBuilder/interface/utils.h"
#include "DataFormats/SiStripDetId/interface/SiStripEnums.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

std::vector<uint32_t> TrackerGeometryUtils::getSiStripDetIds(const GeometricDet& geomDet) {
  std::vector<const GeometricDet*> deepComp;
  geomDet.deepComponents(deepComp);
  std::vector<uint32_t> stripDetIds;
  for (const auto* dep : deepComp) {
    const auto detId = dep->geographicalId();
    if ((detId.subdetId() >= SiStripSubdetector::TIB) && (detId.subdetId() <= SiStripSubdetector::TEC)) {
      stripDetIds.push_back(detId);
    }
  }
  std::stable_sort(
      std::begin(stripDetIds), std::end(stripDetIds), [](DetId a, DetId b) { return a.subdetId() < b.subdetId(); });
  return stripDetIds;
}

std::vector<uint32_t> TrackerGeometryUtils::getOuterTrackerDetIds(const GeometricDet& geomDet) {
  std::vector<const GeometricDet*> deepComp;
  geomDet.deepComponents(deepComp);
  std::vector<uint32_t> OTDetIds;
  for (const auto* dep : deepComp) {
    const auto detId = dep->geographicalId();
    if ((detId.det() == DetId::Detector::Tracker) && (detId.subdetId() > PixelSubdetector::PixelEndcap)) {
      OTDetIds.push_back(detId);
    }
  }
  std::stable_sort(
      std::begin(OTDetIds), std::end(OTDetIds), [](DetId a, DetId b) { return a.subdetId() < b.subdetId(); });
  return OTDetIds;
}
