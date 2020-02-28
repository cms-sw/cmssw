#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTDiscBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTRingBuilder.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <algorithm>

using namespace std;

template <class FilteredView>
void CmsTrackerOTDiscBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsTrackerOTRingBuilder<FilteredView> theCmsTrackerOTRingBuilder;
  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));

  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::panel:
      theCmsTrackerOTRingBuilder.build(fv, subdet, s);
      break;
    default:
      edm::LogError("CmsTrackerOTDiscBuilder")
          << " ERROR - I was expecting a Panel, I got a " << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }
  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerOTDiscBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  switch (det->components().front()->type()) {
    case GeometricDet::panel:
      // nothing to be done because the rings (here named panels) are already sorted ??
      break;
    default:
      edm::LogError("CmsTrackerOTDiscBuilder")
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

template class CmsTrackerOTDiscBuilder<DDFilteredView>;
template class CmsTrackerOTDiscBuilder<cms::DDFilteredView>;
