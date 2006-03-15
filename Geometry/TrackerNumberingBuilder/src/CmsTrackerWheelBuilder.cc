#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerWheelBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerRingBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerPetalBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/precomputed_stable_sort.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<vector>

void CmsTrackerWheelBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, std::string s){
  CmsTrackerRingBuilder theCmsTrackerRingBuilder ;
  CmsTrackerPetalBuilder theCmsTrackerPetalBuilder ;

  GeometricDet * subdet = new GeometricDet(&fv,theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv)));
  switch (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv))){
  case GeometricDet::ring:
    theCmsTrackerRingBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::petal:
    theCmsTrackerPetalBuilder.build(fv,subdet,s);      
    break;
  default:
    edm::LogError("CmsTrackerWheelBuilder")<<" ERROR - I was expecting a Ring or Petal, I got a "<<ExtractStringFromDDD::getString(s,&fv);
  }  
  g->addComponent(subdet);
}

void CmsTrackerWheelBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){
  std::vector< GeometricDet*> comp = det->components();
       
  if(comp.size()){
    if(comp.front()->type()==GeometricDet::petal){
      std::vector< GeometricDet* > compfw;
      std::vector< GeometricDet* > compbw;
      compfw.clear();
      compbw.clear();
      for(uint32_t i=0; i<comp.size();i++){
	if(fabs(comp[i]->translation().z())<fabs(det->translation().z())){
	  compfw.push_back(comp[i]);
	}else{
	  compbw.push_back(comp[i]);      
	}
      }    

      precomputed_stable_sort(compfw.begin(), compfw.end(), ExtractPhiModule());
      precomputed_stable_sort(compbw.begin(), compbw.end(), ExtractPhiModule());

      for(uint32_t i=0; i<compbw.size(); i++){
	compbw[i]->setGeographicalID(DetId(i+1));
      }
      for(uint32_t i=0; i<compfw.size(); i++){
	uint32_t temp = i+1;
	temp|=(1<<7);
	compfw[i]->setGeographicalID(DetId(temp));
      }
      
      det->deleteComponents();
      det->addComponents(compfw);
      det->addComponents(compbw);
      
    }else{
      std::stable_sort(comp.begin(),comp.end(),LessR_module());

      for(uint32_t i=0; i<comp.size(); i++){
	comp[i]->setGeographicalID(DetId(i+1));
      }
      
      det->deleteComponents();
      det->addComponents(comp);
    }
  }else{
    edm::LogError("CmsTrackerWheelBuilder")<<"Where are the Petals or Rings?";
  }
}




