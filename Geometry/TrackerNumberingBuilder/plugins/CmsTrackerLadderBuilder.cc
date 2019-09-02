#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLadderBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include <vector>

template <>
void CmsTrackerLadderBuilder<DDFilteredView>::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s) {
  CmsDetConstruction<DDFilteredView> theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv, g, s);
}

template <>
void CmsTrackerLadderBuilder<DDFilteredView>::sortNS(DDFilteredView& fv, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  //sorting for PhaseI & PhaseII pixel ladder modules
  //sorting also for PhaseII outer tracker rod modules
  std::sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::isLessZ);

  for (uint32_t i = 0; i < comp.size(); i++) {
    det->component(i)->setGeographicalID(i + 1);
  }

  if (comp.empty()) {
    edm::LogError("CmsTrackerLadderBuilder") << "Where are the OT Phase2/ pixel barrel modules modules?";
  }
}
