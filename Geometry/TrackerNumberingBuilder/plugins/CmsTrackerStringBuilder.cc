#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerStringBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

template <class FilteredView>
void CmsTrackerStringBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsDetConstruction<FilteredView> theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv, g, s);
}

template <class FilteredView>
void CmsTrackerStringBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  std::stable_sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessModZ);

  if (!comp.empty()) {
    for (uint32_t i = 0; i < comp.size(); i++) {
      det->component(i)->setGeographicalID(DetId(i + 1));
    }
  } else {
    edm::LogError("CmsTrackerStringBuilder") << "Where are the String's modules?";
  }
}

template class CmsTrackerStringBuilder<DDFilteredView>;
template class CmsTrackerStringBuilder<cms::DDFilteredView>;
