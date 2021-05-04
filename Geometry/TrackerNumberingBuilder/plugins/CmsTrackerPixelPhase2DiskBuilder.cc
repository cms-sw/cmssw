#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2DiskBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2RingBuilder.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <algorithm>

using namespace std;

template <class FilteredView>
void CmsTrackerPixelPhase2DiskBuilder<FilteredView>::buildComponent(FilteredView& fv,
                                                                    GeometricDet* g,
                                                                    const std::string& s) {
  CmsTrackerPixelPhase2RingBuilder<FilteredView> theCmsTrackerPixelPhase2RingBuilder;
  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));

  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::panel:
      theCmsTrackerPixelPhase2RingBuilder.build(fv, subdet, s);
      break;
    default:
      edm::LogError("CmsTrackerPixelPhase2DiskBuilder")
          << " ERROR - I was expecting a Panel, I got a " << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }
  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerPixelPhase2DiskBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  switch (det->components().front()->type()) {
    case GeometricDet::panel:
      // nothing to be done because the rings (here named panels) are already sorted ??
      break;
    default:
      edm::LogError("CmsTrackerPixelPhase2Builder")
          << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();
  }

  GeometricDet::GeometricDetContainer rings;
  uint32_t totalrings = comp.size();

  for (uint32_t rn = 0; rn < totalrings; rn++) {
    rings.emplace_back(det->component(rn));
    uint32_t blade = rn + 1;
    uint32_t panel = 1;
    uint32_t temp = (blade << 2) | panel;
    rings[rn]->setGeographicalID(temp);
  }

  det->clearComponents();
  det->addComponents(rings);
}

template class CmsTrackerPixelPhase2DiskBuilder<DDFilteredView>;
template class CmsTrackerPixelPhase2DiskBuilder<cms::DDFilteredView>;
