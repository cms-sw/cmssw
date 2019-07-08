#include "Geometry/TrackerNumberingBuilder/interface/utils.h"
#include "DataFormats/TrackerCommon/interface/SiStripEnums.h"

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
