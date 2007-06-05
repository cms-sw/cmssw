#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPanelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>


void CmsTrackerPanelBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){

  CmsDetConstruction theCmsDetConstruction;
  switch (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv))){
  case GeometricDet::DetUnit:
           theCmsDetConstruction.buildComponent(fv,g,s);
    break;
  default:
    edm::LogError("CmsTrackerPanelBuilder")<<" ERROR - I was expecting a Plaq, I got a "<<ExtractStringFromDDD::getString(s,&fv);
    ;
  }  
}

void CmsTrackerPanelBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
 GeometricDet::GeometricDetContainer comp = det->components();

 switch(det->components().front()->type()){
 case GeometricDet::DetUnit: std::stable_sort(comp.begin(),comp.end(),LessR()); break;
 default:
   edm::LogError("CmsTrackerPanelBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type(); 
 }
  for(uint32_t i=0; i<comp.size();i++){
    comp[i]->setGeographicalID(DetId(i+1));
  } 
 
  det->deleteComponents();
  det->addComponents(comp);
}
