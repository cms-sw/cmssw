#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAModules_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAModules_h

#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits {

  // The OT modules of the CA, the P sensors of the PS modules in the OT barrel, in TrackerGeometry::detUnits() order:
  // the CA module index is the position in this list
  inline std::vector<GeomDetUnit const*> caModules(TrackerGeometry const& geometry) {
    std::vector<GeomDetUnit const*> modules;
    for (auto const* detUnit : geometry.detUnits()) {
      const DetId detId = detUnit->geographicalId();
      if (geometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP &&
          detId.subdetId() == StripSubdetector::TOB)
        modules.push_back(detUnit);
    }
    return modules;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAModules_h
