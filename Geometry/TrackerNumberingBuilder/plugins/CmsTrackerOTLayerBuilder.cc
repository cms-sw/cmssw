#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTLayerBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTRingBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLadderBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/TrackerStablePhiSort.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

void CmsTrackerOTLayerBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){

  LogTrace("DetConstruction") << " CmsTrackerOTLayerBuilder::buildComponent ";
  CmsTrackerLadderBuilder theCmsTrackerLadderBuilder;
  CmsTrackerOTRingBuilder theCmsTrackerOTRingBuilder;

  GeometricDet * subdet = new GeometricDet(&fv,theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv)));
  switch (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv))){
  case GeometricDet::ladder:
    theCmsTrackerLadderBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::panel:
    theCmsTrackerOTRingBuilder.build( fv, subdet, s );
    break;
  default:
    edm::LogError("CmsTrackerOTLayerBuilder")<<" ERROR - I was expecting a ladder or a panel, I got a "<<ExtractStringFromDDD::getString(s,&fv);
  }  
  g->addComponent(subdet);

}

void CmsTrackerOTLayerBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){

  GeometricDet::ConstGeometricDetContainer comp = det->components();

  //order ladder and rings together
  GeometricDet::GeometricDetContainer rods;
  GeometricDet::GeometricDetContainer ringsNeg;
  GeometricDet::GeometricDetContainer ringsPos;
  rods.clear();
  ringsNeg.clear();
  ringsPos.clear();

  for(uint32_t i=0; i<comp.size();i++){
    auto component = det->component(i);
    if(component->type()== GeometricDet::ladder){
      rods.push_back(component);
    } else if(component->type()== GeometricDet::panel){
      if(component->translation().z() < 0.){
        ringsNeg.push_back(component);
      } else if (component->translation().z() > 0.) {
        ringsPos.push_back(component);
      }
    } else {
      edm::LogError("CmsTrackerOTLayerBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type();
    }
  }
      
  // negative rings 
  if(ringsNeg.size() != 0){
    std::sort(ringsNeg.begin(),ringsNeg.end(),LessZ());  
    uint32_t  totalringsNeg = ringsNeg.size();
  
    LogTrace("DetConstruction") << " Neg rings ordered by z: ";
    for ( uint32_t ring = 0; ring < totalringsNeg; ring++) {
      uint32_t temp = ring+1;
      temp|=(1<<8);
      ringsNeg[ring]->setGeographicalID(temp);
      LogTrace("BuildingTrackerDetId") << "\t\t\t DetId >> " << temp << "(r: " << sqrt(ringsNeg[ring]->translation().Perp2()) << ", phi: " << ringsNeg[ring]->phi() << ", z: " << ringsNeg[ring]->translation().z() << ")";
    }
  }

  // rods 
  if(rods.size() != 0){
    TrackerStablePhiSort(rods.begin(), rods.end(), ExtractPhi());
    uint32_t  totalrods = rods.size();
  
    LogTrace("DetConstruction") << " Rods ordered by phi: ";
    for ( uint32_t rod = 0; rod < totalrods; rod++) {
      uint32_t temp = rod+1;
      temp|=(3<<8);
      rods[rod]->setGeographicalID(DetId(temp));
      LogTrace("BuildingTrackerDetId") << "\t\t\t DetId >> " << temp << "(r: " << sqrt(rods[rod]->translation().Perp2()) << ", phi: " << rods[rod]->phi() << ", z: " << rods[rod]->translation().z() << ")";
    }
  }

  // positive rings 
  if(ringsPos.size() != 0){
    std::sort(ringsPos.begin(),ringsPos.end(),LessZ());  
    uint32_t  totalringsPos = ringsPos.size();
  
  
    LogTrace("DetConstruction") << " Pos rings ordered by z: ";
    for ( uint32_t ring = 0; ring < totalringsPos; ring++) {
      uint32_t temp = ring+1;
      temp|=(2<<8);
      ringsPos[ring]->setGeographicalID(temp);
      LogTrace("BuildingTrackerDetId") << "\t\t\t DetId >> " << temp << "(r: " << sqrt(ringsPos[ring]->translation().Perp2()) << ", phi: " << ringsPos[ring]->phi() << ", z: " << ringsPos[ring]->translation().z() << ")";
    }
  }

  det->clearComponents();
  det->addComponents(ringsNeg);
  det->addComponents(rods);
  det->addComponents(ringsPos);

}

