#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPanelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

#include "Geometry/TrackerNumberingBuilder/plugins/TrackerStablePhiSort.h"

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
 GeometricDet::GeometricDetContainer & comp = det->components();

 if (comp.front()->type()==GeometricDet::DetUnit){ 

   // NP** Phase 2 Sort Modules within Rings

   std::string comp_name = comp[0]->name();
   if( !(comp_name.find("PixelForwardDisk") < comp_name.size())  ) {

     //   if( fabs( comp[0]->translation().z() ) > 1000 ) { now commented by AT+NP
     //std::cerr<<"PHASE 2!!!"<<std::endl;
     TrackerStablePhiSort(comp.begin(), comp.end(), ExtractPhi());
     stable_sort(comp.begin(), comp.end() ,PhiSortNP());
   }
   else
     // original one
     std::sort(comp.begin(),comp.end(),LessR());
 }
 else
   edm::LogError("CmsTrackerPanelBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type(); 

  for(uint32_t i=0; i<comp.size();i++){
    comp[i]->setGeographicalID(i+1);
  } 
 
}
