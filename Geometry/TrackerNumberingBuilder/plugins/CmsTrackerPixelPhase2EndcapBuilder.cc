#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2EndcapBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPhase2TPDiskBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2DiskBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTDiscBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include <vector>

#include <bitset>

template <class T>
void CmsTrackerPixelPhase2EndcapBuilder<T>::buildComponent(T& fv, GeometricDet* g, std::string s) {
  CmsTrackerPhase2TPDiskBuilder<T> theCmsTrackerPhase2DiskBuilder;
  CmsTrackerPixelPhase2DiskBuilder<T> theCmsTrackerPixelPhase2DiskBuilder;
  CmsTrackerOTDiscBuilder<T> theCmsTrackerOTDiscBuilder;

  GeometricDet* subdet = new GeometricDet(
      &fv, CmsTrackerLevelBuilder<T>::theCmsTrackerStringToEnum.type(ExtractStringFromDDD<T>::getString(s, &fv)));
  switch (CmsTrackerLevelBuilder<T>::theCmsTrackerStringToEnum.type(ExtractStringFromDDD<T>::getString(s, &fv))) {
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
          << " ERROR - I was expecting a Disk... I got a " << ExtractStringFromDDD<T>::getString(s, &fv);
  }

  g->addComponent(subdet);
}

template <class T>
void CmsTrackerPixelPhase2EndcapBuilder<T>::sortNS(T& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessModZ);

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(
        i + 1);  // Every subdetector: Inner pixel first, OT later, then sort by disk number
  }
}

template class CmsTrackerPixelPhase2EndcapBuilder<DDFilteredView>;
template class CmsTrackerPixelPhase2EndcapBuilder<cms::DDFilteredView>;
