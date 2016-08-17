#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2RingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

#include "Geometry/TrackerNumberingBuilder/plugins/TrackerStablePhiSort.h"

void CmsTrackerPixelPhase2RingBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){
  //std::cout << " Sono in CmsTrackerPixelPhase2RingBuilder " << ExtractStringFromDDD::getString( s, &fv ) << std::endl;
  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);

}

void CmsTrackerPixelPhase2RingBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){

  GeometricDet::ConstGeometricDetContainer & comp = det->components();

  //std::cout << "Sono in CmsTrackerPixelPhase2RingBuilder: comp.size() " << comp.size() << std::endl;
  //increasing phi taking into account the sub-modules

  TrackerStablePhiSort(comp.begin(), comp.end(), ExtractPhi());


 /* for(uint32_t i=0; i<comp.size();i++){
    det->component(i)->setGeographicalID(i+1);
    std::cout << "Phase2RingBuilder: z " << comp[i]->translation().z() << std::endl;
    std::cout << "Phase2RingBuilder: phi " << comp[i]->translation().phi() << std::endl;
  } */
 
}
