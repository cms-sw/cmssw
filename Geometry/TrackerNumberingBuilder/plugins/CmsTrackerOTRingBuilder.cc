#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTRingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"

#include <vector>

template <class T>
void CmsTrackerOTRingBuilder<T>::buildComponent(T& fv, GeometricDet* g, std::string s) {
  CmsDetConstruction<T> theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv, g, s);
}

template <class T>
void CmsTrackerOTRingBuilder<T>::sortNS(T& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  //increasing phi taking into account the sub-modules
  trackerStablePhiSort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::getPhiGluedModule);

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(i + 1);
  }
}

template class CmsTrackerOTRingBuilder<DDFilteredView>;
template class CmsTrackerOTRingBuilder<cms::DDFilteredView>;
