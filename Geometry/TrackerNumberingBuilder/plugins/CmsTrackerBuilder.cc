#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerSubStrctBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase1EndcapBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2EndcapBuilder.h"

#include <bitset>

template <class FilteredView>
void CmsTrackerBuilder<FilteredView>::buildComponent(FilteredView& fv, GeometricDet* g, const std::string& s) {
  CmsTrackerSubStrctBuilder<FilteredView> theCmsTrackerSubStrctBuilder;
  CmsTrackerPixelPhase1EndcapBuilder<FilteredView> theCmsTrackerPixelPhase1EndcapBuilder;
  CmsTrackerPixelPhase2EndcapBuilder<FilteredView> theCmsTrackerPixelPhase2EndcapBuilder;

  GeometricDet* subdet = new GeometricDet(&fv,
                                          CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
                                              ExtractStringFromDDD<FilteredView>::getString(s, &fv)));
  switch (CmsTrackerLevelBuilder<FilteredView>::theCmsTrackerStringToEnum.type(
      ExtractStringFromDDD<FilteredView>::getString(s, &fv))) {
    case GeometricDet::PixelBarrel:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelPhase1Barrel:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelPhase2Barrel:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelEndCap:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelPhase1EndCap:
      theCmsTrackerPixelPhase1EndcapBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::PixelPhase2EndCap:
      theCmsTrackerPixelPhase2EndcapBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::OTPhase2EndCap:
      theCmsTrackerPixelPhase2EndcapBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::TIB:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::TOB:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::OTPhase2Barrel:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::TEC:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    case GeometricDet::TID:
      theCmsTrackerSubStrctBuilder.build(fv, subdet, s);
      break;
    default:
      edm::LogError("CmsTrackerBuilder") << " ERROR - I was expecting a SubDet, I got a "
                                         << ExtractStringFromDDD<FilteredView>::getString(s, &fv);
  }

  g->addComponent(subdet);
}

template <class FilteredView>
void CmsTrackerBuilder<FilteredView>::sortNS(FilteredView&, GeometricDet* det) {
  GeometricDet::ConstGeometricDetContainer& comp = det->components();
  std::stable_sort(comp.begin(), comp.end(), CmsTrackerLevelBuilderHelper::subDetByType);

  for (uint32_t i = 0; i < comp.size(); i++) {
    uint32_t temp = comp[i]->type();
    det->component(i)->setGeographicalID(
        temp %
        100);  // it relies on the fact that the GeometricDet::GDEnumType enumerators used to identify the subdetectors in the upgrade geometries are equal to the ones of the present detector + n*100
  }
}

template class CmsTrackerBuilder<DDFilteredView>;
template class CmsTrackerBuilder<cms::DDFilteredView>;
