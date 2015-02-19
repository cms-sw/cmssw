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
  switch (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv))){
  case GeometricDet::DetUnit:
           theCmsDetConstruction.buildComponent(fv,g,s);
    break;
  default:
    edm::LogError("CmsTrackerOTRingBuilder")<<" ERROR - I was expecting a Plaq, I got a "<<ExtractStringFromDDD::getString(s,&fv);
    ;
  }  
}

void CmsTrackerOTRingBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
 GeometricDet::ConstGeometricDetContainer & comp = det->components();

 if (comp.front()->type()==GeometricDet::DetUnit){ 

   TrackerStablePhiSort(comp.begin(), comp.end(), ExtractPhi());
   stable_sort(comp.begin(), comp.end() ,PhiSortNP());
   
 }
 else
   edm::LogError("CmsTrackerOTRingBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type(); 

  for(uint32_t i=0; i<comp.size();i++){
    det->component(i)->setGeographicalID(i+1);
  } 
 
}
