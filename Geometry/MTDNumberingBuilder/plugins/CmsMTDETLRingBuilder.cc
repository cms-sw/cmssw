#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDETLRingBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDConstruction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>

#include "Geometry/MTDNumberingBuilder/plugins/MTDStablePhiSort.h"

void CmsMTDETLRingBuilder::buildComponent(DDFilteredView& fv, GeometricTimingDet* g, std::string s){

  CmsMTDConstruction theCmsMTDConstruction;
  theCmsMTDConstruction.buildComponent(fv,g,s);
  
}

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

void CmsMTDETLRingBuilder::sortNS(DDFilteredView& fv, GeometricTimingDet* det){

  GeometricTimingDet::ConstGeometricTimingDetContainer & comp = det->components();

  //increasing phi taking into account the sub-modules
  MTDStablePhiSort(comp.begin(), comp.end(), ExtractPhiGluedModule());
  
}
