#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDModuleBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDConstruction.h"
#include <vector>

void CmsMTDModuleBuilder::buildComponent(DDFilteredView& fv, GeometricTimingDet* g, std::string side){
  
  CmsMTDConstruction theCmsMTDConstruction;
  theCmsMTDConstruction.buildComponent(fv,g,side);  
}


#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
void CmsMTDModuleBuilder::sortNS(DDFilteredView& fv, GeometricTimingDet* det){
  GeometricTimingDet::ConstGeometricTimingDetContainer & comp = det->components();

  std::stable_sort(comp.begin(),comp.end(),isLessZ);

  if (comp.empty() ){
   edm::LogError("CmsMTDModuleBuilder") << "Where are the ETL modules?";
  } 

}




