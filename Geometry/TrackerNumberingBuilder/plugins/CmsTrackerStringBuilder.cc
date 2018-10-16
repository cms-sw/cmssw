#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerStringBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsDetConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>


void CmsTrackerStringBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){
  CmsDetConstruction theCmsDetConstruction;
  theCmsDetConstruction.buildComponent(fv,g,s);  
}

void CmsTrackerStringBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  std::stable_sort(comp.begin(),comp.end(),isLessModZ);
  
  if(!comp.empty()){
    for(uint32_t i=0; i<comp.size();i++){
      det->component(i)->setGeographicalID(DetId(i+1));
    }
  }else{
       edm::LogError("CmsTrackerStringBuilder")<<"Where are the String's modules?";
  }
}


