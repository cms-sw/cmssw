#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2EndcapBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPhase2TPDiskBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2DiskBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTDiscBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

template<>
void CmsTrackerPixelPhase2EndcapBuilder<DDFilteredView>::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s) {
  CmsTrackerPhase2TPDiskBuilder<DDFilteredView> theCmsTrackerPhase2DiskBuilder;
  CmsTrackerPixelPhase2DiskBuilder<DDFilteredView> theCmsTrackerPixelPhase2DiskBuilder;
  CmsTrackerOTDiscBuilder<DDFilteredView> theCmsTrackerOTDiscBuilder;

  GeometricDet* subdet = new GeometricDet(&fv, theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s, &fv)));
  switch (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s, &fv))) {
    case GeometricDet::PixelPhase2FullDisk:
      theCmsTrackerPhase2DiskBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelPhase2ReducedDisk:
      theCmsTrackerPhase2DiskBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelPhase2TDRDisk:
      theCmsTrackerPixelPhase2DiskBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::OTPhase2Wheel:
      theCmsTrackerOTDiscBuilder.build(fv, subdet, s);
      break;

    default:
      edm::LogError("CmsTrackerPixelPhase2EndcapBuilder")
          << " ERROR - I was expecting a Disk... I got a " << ExtractStringFromDDD::getString(s, &fv);
  }

  g->addComponent(subdet);
}

template<>
void CmsTrackerPixelPhase2EndcapBuilder<DDFilteredView>::sortNS(DDFilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessModZ);

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(
        i + 1);  // Every subdetector: Inner pixel first, OT later, then sort by disk number
  }
}
