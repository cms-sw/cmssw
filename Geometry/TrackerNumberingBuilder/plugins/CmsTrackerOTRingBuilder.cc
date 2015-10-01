#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTRingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

#include "Geometry/TrackerNumberingBuilder/plugins/TrackerStablePhiSort.h"

void CmsTrackerOTRingBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){

  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);

}

void CmsTrackerOTRingBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){

  GeometricDet::GeometricDetContainer & comp = det->components();

  //increasing phi and modules in the same pt module are sorted by increasing abs(z)
  TrackerStablePhiSort(comp.begin(), comp.end(), ExtractPhiGluedModule());
  stable_sort(comp.begin(), comp.end() ,PhiSortNP());

  for(uint32_t i=0; i<comp.size();i++){
    comp[i]->setGeographicalID(i+1);
  } 
 
}
