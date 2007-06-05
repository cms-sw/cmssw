#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLadderBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include <vector>

void CmsTrackerLadderBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){

  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);  
}

void CmsTrackerLadderBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
  GeometricDet::GeometricDetContainer comp = det->components();

 switch(det->components().front()->type()){
 case GeometricDet::DetUnit: std::stable_sort(comp.begin(),comp.end(),LessZ()); break;	
 default:
   edm::LogError("CmsTrackerLadderBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type(); 
 }
 
  for(uint32_t i=0; i<comp.size();i++){
    comp[i]->setGeographicalID(DetId(i+1));
  } 

  det->deleteComponents();
  det->addComponents(comp);
 
}




