#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerRingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsDetConstruction.h"
#include "Geometry/TrackerNumberingBuilder/interface/precomputed_stable_sort.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>


void CmsTrackerRingBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){
  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);  
}

void CmsTrackerRingBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
  std::vector< GeometricDet* > comp = det->components();
  fv.firstChild(); 
  std::vector< GeometricDet* > compfw;
  std::vector< GeometricDet* > compbw;
  compfw.clear();
  compbw.clear();


  switch(det->components().front()->type()){

  case GeometricDet::mergedDet: precomputed_stable_sort(comp.begin(), comp.end(), ExtractPhi()); break;
  case GeometricDet::DetUnit:precomputed_stable_sort(comp.begin(), comp.end(), ExtractPhi()); break;
  default:
    edm::LogError("CmsTrackerRingBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type(); 
  }

  std::string part="TkDDDStructure";
  if(ExtractStringFromDDD::getString(part,&fv) == "TECGluedDet"||ExtractStringFromDDD::getString(part,&fv) == "TECDet"){
    
    for(uint32_t i=0; i<comp.size();i++){
      uint32_t temp = i+1;
      comp[i]->setGeographicalID(DetId(temp));
    }
    
    det->deleteComponents();
    det->addComponents(compfw);
    det->addComponents(compbw);
    
  }else{
    for(uint32_t i=0; i<comp.size();i++){
      if(fabs(comp[i]->translation().z())<fabs(det->translation().z())){      
	compfw.push_back(comp[i]);
      }else{
	compbw.push_back(comp[i]);
      }
    }
    
    if(compbw.size()){
      for(uint32_t i=0; i<compbw.size();i++){
	uint32_t temp = i+1;
	compbw[i]->setGeographicalID(DetId(temp));
      }
    }
    
    if(compfw.size()){
      for(uint32_t i=0; i<compfw.size();i++){
	uint32_t temp = i+1;
	temp |=(1<<5);
	compfw[i]->setGeographicalID(DetId(temp));
      }
    }
    
    det->deleteComponents();
    det->addComponents(compfw);
    det->addComponents(compbw);
    
  }
  
  fv.parent();
  
}





