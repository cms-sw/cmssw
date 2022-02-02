#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2DoubleDiskBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2SubDiskBuilder.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <algorithm>

using namespace std;

template <class FilteredView>
void CmsTrackerPixelPhase2DoubleDiskBuilder<FilteredView>::buildComponent(FilteredView& fv,
                                                                          GeometricDet* g,
                                                                          const std::string& s) {
  CmsTrackerPixelPhase2SubDiskBuilder<FilteredView> theCmsTrackerPixelPhase2SubDiskBuilder;
  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));

  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::PixelPhase2SubDisk:
      theCmsTrackerPixelPhase2SubDiskBuilder.build(fv, subdet, s);
      break;
    default:
      edm::LogError("CmsTrackerPixelPhase2DoubleDiskBuilder")
          << " ERROR - I was expecting a SubDisk, I got a " << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }
  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerPixelPhase2DoubleDiskBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessModZ);

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(i);
  }
}

template class CmsTrackerPixelPhase2DoubleDiskBuilder<DDFilteredView>;
template class CmsTrackerPixelPhase2DoubleDiskBuilder<cms::DDFilteredView>;
