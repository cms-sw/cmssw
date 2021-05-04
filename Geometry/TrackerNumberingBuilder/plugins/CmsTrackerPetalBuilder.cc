#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPetalBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerRingBuilder.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

template <class FilteredView>
void CmsTrackerPetalBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  GeometricDet* det = new GeometricDet(&fv,
                                       CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                           ExtractStringFromDDD<FilteredView>::getString(s, &fv)));
  CmsTrackerRingBuilder<FilteredView> theCmsTrackerRingBuilder;
  theCmsTrackerRingBuilder.build(fv, det, s);
  g->addComponent(det);
}

template <class FilteredView>
void CmsTrackerPetalBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  if (comp.front()->type() == GeometricDet::ring)
    std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessRModule);
  else
    edm::LogError("CmsTrackerPetalBuilder")
        << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();

  // Maximum Number fo TEC Rings is 7 in order
  // to discover from which number we have to start
  // the operation is MaxRing - RealRingNumber + 1 (C++)

  uint32_t startring = 8 - comp.size();

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(startring + i);
  }
}

template class CmsTrackerPetalBuilder<DDFilteredView>;
template class CmsTrackerPetalBuilder<cms::DDFilteredView>;
