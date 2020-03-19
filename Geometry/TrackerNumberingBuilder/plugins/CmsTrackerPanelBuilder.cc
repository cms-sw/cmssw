#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPanelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

template <class FilteredView>
void CmsTrackerPanelBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsDetConstruction<FilteredView> theCmsDetConstruction;
  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::DetUnit:
      theCmsDetConstruction.buildComponent(fv, g, s);
      break;
    default:
      edm::LogError("CmsTrackerPanelBuilder")
          << " ERROR - I was expecting a Plaq, I got a " << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
      ;
  }
}

template <class FilteredView>
void CmsTrackerPanelBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  if (comp.front()->type() == GeometricDet::DetUnit)
    std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessR);
  else
    edm::LogError("CmsTrackerPanelBuilder")
        << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(i + 1);
  }
}

template class CmsTrackerPanelBuilder<DDFilteredView>;
template class CmsTrackerPanelBuilder<cms::DDFilteredView>;
