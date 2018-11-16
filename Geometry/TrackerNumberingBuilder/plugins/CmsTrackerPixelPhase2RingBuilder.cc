#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2RingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"

#include <vector>

void CmsTrackerPixelPhase2RingBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){
  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);

}

void CmsTrackerPixelPhase2RingBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){

  GeometricDet::ConstGeometricDetContainer & comp = det->components();

   //increasing phi taking into account the sub-modules

  trackerStablePhiSort(comp.begin(), comp.end(), getPhi);


  for(uint32_t i=0; i<comp.size();i++){
    det->component(i)->setGeographicalID(i+1);
  } 
 
}
