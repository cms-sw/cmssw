#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDTrayBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDModuleBuilder.h"
#include "Geometry/MTDNumberingBuilder/plugins/MTDStablePhiSort.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

void CmsMTDTrayBuilder::buildComponent(DDFilteredView& fv, GeometricTimingDet* g, std::string side){
  
  CmsMTDModuleBuilder theCmsMTDModuleBuilder;
  
  GeometricTimingDet * subdet = new GeometricTimingDet(&fv,theCmsMTDStringToEnum.type(fv.logicalPart().name().fullname()));
  switch (theCmsMTDStringToEnum.type(fv.logicalPart().name().fullname())){
  case GeometricTimingDet::BTLTray:
    theCmsMTDModuleBuilder.build(fv,subdet,side);      
    break;  
  default:
    throw cms::Exception("CmsMTDTrayBuilder")<<" ERROR - I was expecting a Tray, I got a "<< fv.logicalPart().name().fullname();
  }  
  
  g->addComponent(subdet);
}

void CmsMTDTrayBuilder::sortNS(DDFilteredView& fv, GeometricTimingDet* det){

  GeometricTimingDet::ConstGeometricTimingDetContainer comp = det->components();

  //order ladder and rings together
  GeometricTimingDet::GeometricTimingDetContainer rods;
  rods.clear();
  

  for(uint32_t i=0; i<comp.size();i++){
    auto component = det->component(i);
    if(component->type()== GeometricTimingDet::BTLTray){
      rods.emplace_back(component);
    } else {
      edm::LogError("CmsMTDOTLayerBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type();
    }
  }
  
  // rods 
  if(!rods.empty()){
    MTDStablePhiSort(rods.begin(), rods.end(), ExtractPhi());
    uint32_t  totalrods = rods.size();
  
    LogTrace("DetConstruction") << " Rods ordered by phi: ";
    for ( uint32_t rod = 0; rod < totalrods; rod++) {
      uint32_t temp = rod+1;
      temp|=(3<<8);
      rods[rod]->setGeographicalID(DetId(temp));
      LogTrace("BuildingMTDDetId") << "\t\t\t DetId >> " << temp << "(r: " << sqrt(rods[rod]->translation().Perp2()) << ", phi: " << rods[rod]->phi() << ", z: " << rods[rod]->translation().z() << ")";
    }
  }
  
  det->clearComponents();
  det->addComponents(rods);

}

