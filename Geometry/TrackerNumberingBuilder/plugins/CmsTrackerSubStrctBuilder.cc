#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerSubStrctBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLayerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTLayerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerWheelBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDiskBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include <vector>

#include <bitset>

template <class FilteredView>
void CmsTrackerSubStrctBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsTrackerLayerBuilder<FilteredView> theCmsTrackerLayerBuilder;
  CmsTrackerOTLayerBuilder<FilteredView> theCmsTrackerOTLayerBuilder;
  CmsTrackerWheelBuilder<FilteredView> theCmsTrackerWheelBuilder;
  CmsTrackerDiskBuilder<FilteredView> theCmsTrackerDiskBuilder;

  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));
  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::layer:
      theCmsTrackerLayerBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::OTPhase2Layer:
      theCmsTrackerOTLayerBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::wheel:
      theCmsTrackerWheelBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::disk:
      theCmsTrackerDiskBuilder.build(fv, subdet, s);
      break;

    default:
      edm::LogError("CmsTrackerSubStrctBuilder") << " ERROR - I was expecting a Layer ,Wheel or Disk... I got a "
                                                 << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }

  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerSubStrctBuilder<FilteredView>::sortNS(FilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  switch (comp.front()->type()) {
    case GeometricDet::layer:
      std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessR);
      break;
    case GeometricDet::OTPhase2Layer:
      std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessR);
      break;
    case GeometricDet::wheel:
      std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessModZ);
      break;
    case GeometricDet::disk:
      std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessModZ);
      break;
    default:
      edm::LogError("CmsTrackerSubStrctBuilder")
          << "ERROR - wrong SubDet to sort..... " << det->components().front()->type();
  }

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(i + 1);  // Every subdetector: Layer/Disk/Wheel Number
  }
}

template class CmsTrackerSubStrctBuilder<DDFilteredView>;
template class CmsTrackerSubStrctBuilder<cms::DDFilteredView>;
