#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerWheelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerRingBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPetalBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <bitset>

template <class FilteredView>
void CmsTrackerWheelBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsTrackerRingBuilder<FilteredView> theCmsTrackerRingBuilder;
  CmsTrackerPetalBuilder<FilteredView> theCmsTrackerPetalBuilder;

  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));
  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::ring:
      theCmsTrackerRingBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::petal:
      theCmsTrackerPetalBuilder.build(fv, subdet, s);
      break;
    default:
      edm::LogError("CmsTrackerWheelBuilder") << " ERROR - I was expecting a Ring or Petal, I got a "
                                              << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }
  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerWheelBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  if (!comp.empty()) {
    if (comp.front()->type() == GeometricDet::petal) {
      GeometricDet::GeometricDetContainer compfw;
      GeometricDet::GeometricDetContainer compbw;
      compfw.clear();
      compbw.clear();
      for (uint32_t i = 0; i < comp.size(); i++) {
        if (std::abs(comp[i]->translation().z()) < std::abs(det->translation().z())) {
          compfw.emplace_back(det->component(i));
        } else {
          compbw.emplace_back(det->component(i));
        }
      }

      trackerStablePhiSort(compfw.begin(), compfw.end(), CmsTrackerLevelBuilderHelper::getPhiModule);
      trackerStablePhiSort(compbw.begin(), compbw.end(), CmsTrackerLevelBuilderHelper::getPhiModule);

      //
      // TEC
      // Wheel Part:   3 bits [back:1 front:2]
      // Petal Number: 4 bits [1,...,8]
      //
      for (uint32_t i = 0; i < compbw.size(); i++) {
        uint32_t temp = i + 1;
        temp |= (1 << 4);
        compbw[i]->setGeographicalID(DetId(temp));
      }
      for (uint32_t i = 0; i < compfw.size(); i++) {
        uint32_t temp = i + 1;
        temp |= (2 << 4);
        compfw[i]->setGeographicalID(DetId(temp));
      }

      det->clearComponents();
      det->addComponents(compfw);
      det->addComponents(compbw);

    } else {
      std::stable_sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessRModule);

      // TID
      // Disk Number: 2 bits [1,2,3]
      for (uint32_t i = 0; i < comp.size(); i++) {
        det->component(i)->setGeographicalID(DetId(i + 1));
      }
    }
  } else {
    edm::LogError("CmsTrackerWheelBuilder") << "Where are the Petals or Rings?";
  }
}

template class CmsTrackerWheelBuilder<DDFilteredView>;
template class CmsTrackerWheelBuilder<cms::DDFilteredView>;
