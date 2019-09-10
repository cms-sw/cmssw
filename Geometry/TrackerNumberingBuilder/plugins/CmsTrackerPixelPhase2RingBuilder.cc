#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2RingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"

#include <vector>

template <class FilteredView>
void CmsTrackerPixelPhase2RingBuilder<FilteredView>::buildComponent(FilteredView& fv,
                                                                    GeometricDet* g,
                                                                    const std::string& s) {
  CmsDetConstruction<FilteredView> theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv, g, s);
}

template <class FilteredView>
void CmsTrackerPixelPhase2RingBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  //increasing phi taking into account the sub-modules

  trackerStablePhiSort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::getPhi);

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(i + 1);
  }
}

template class CmsTrackerPixelPhase2RingBuilder<DDFilteredView>;
template class CmsTrackerPixelPhase2RingBuilder<cms::DDFilteredView>;
