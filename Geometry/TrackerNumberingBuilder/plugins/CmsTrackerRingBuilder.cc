#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerRingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <bitset>

void CmsTrackerRingBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){
  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);  
}

void CmsTrackerRingBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
  GeometricDet::ConstGeometricDetContainer & comp = det->components();
  fv.firstChild(); 
  GeometricDet::GeometricDetContainer compfw;
  GeometricDet::GeometricDetContainer compbw;


  switch(comp.front()->type()) {
  
  case GeometricDet::mergedDet: 
    trackerStablePhiSort(comp.begin(), comp.end(), getPhiGluedModule);
    break;
  case GeometricDet::DetUnit: 
    trackerStablePhiSort(comp.begin(), comp.end(), getPhi);
    break;
  default:
    edm::LogError("CmsTrackerRingBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type(); 
  }
  
  static std::string const part("TkDDDStructure");
  static std::string const TECGluedDet("TECGluedDet");
  static std::string const TECDet("TECDet");

  std::string const pname = ExtractStringFromDDD::getString(part,&fv); 

  // TEC
  // Module Number: 3 bits [1,...,5 at most]
  if(pname== TECGluedDet || pname == TECDet){
    
    // TEC- 
    if( det->translation().z() < 0 && pname == TECDet) {
      trackerStablePhiSort(comp.begin(), comp.end(), getPhiMirror);
    }
    
    if( det->translation().z() < 0 && pname == TECGluedDet) {
      trackerStablePhiSort(comp.begin(), comp.end(), getPhiGluedModuleMirror);
    }

    for(uint32_t i=0; i<comp.size();i++)
      det->component(i)->setGeographicalID(i+1);

  } else {
    // TID
    // Ring Side: 2 bits [back:1 front:2]
    // Module Number: 5 bits [1,...,20 at most]
    //
    for(uint32_t i=0; i<comp.size();i++){
      if(std::abs(comp[i]->translation().z())<std::abs(det->translation().z())){      
	compfw.emplace_back(det->component(i));
      } else {
	compbw.emplace_back(det->component(i));
      }
    }
    
    for(uint32_t i=0; i<compbw.size();i++){
      uint32_t temp = i+1;
      temp |=(1<<5);
      compbw[i]->setGeographicalID(temp);
    }
    
    for(uint32_t i=0; i<compfw.size();i++){
      uint32_t temp = i+1;
      temp |=(2<<5);
      compfw[i]->setGeographicalID(temp);
    }
    
    det->clearComponents();
    det->addComponents(compfw);
    det->addComponents(compbw);
    
  }
  
  fv.parent();
  
}





