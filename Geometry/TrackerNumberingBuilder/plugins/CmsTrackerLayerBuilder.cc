#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLayerBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerStringBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerRodBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLadderBuilder.h"
#include "Geometry/TrackerNumberingBuilder/interface/trackerStablePhiSort.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <bitset>

void CmsTrackerLayerBuilder::buildComponent(DDFilteredView& fv, GeometricDet* g, const std::string s){

  CmsTrackerStringBuilder theCmsTrackerStringBuilder ;
  CmsTrackerRodBuilder theCmsTrackerRodBuilder;
  CmsTrackerLadderBuilder theCmsTrackerLadderBuilder;

  GeometricDet * subdet = new GeometricDet(&fv,theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv)));
  switch (theCmsTrackerStringToEnum.type(ExtractStringFromDDD::getString(s,&fv))){
  case GeometricDet::strng:
    theCmsTrackerStringBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::rod:
    theCmsTrackerRodBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::ladder:
    theCmsTrackerLadderBuilder.build(fv,subdet,s);      
    break;
  default:
    edm::LogError("CmsTrackerLayerBuilder")<<" ERROR - I was expecting a String, Rod or Ladder, I got a "<<ExtractStringFromDDD::getString(s,&fv);

  }  
  g->addComponent(subdet);

}

void CmsTrackerLayerBuilder::sortNS(DDFilteredView& fv, GeometricDet* det){

  GeometricDet::ConstGeometricDetContainer& comp = det->components();

  // TIB
  // SubDetector Side: 2 bits [TIB-:1 TIB+:2]
  // Layer Part      : 2 bits [internal:1 external:0]
  // String Number   : 6 bits [1,...,56 (at most)]
  //
  if(comp.front()->type()== GeometricDet::strng){
    float layerRadius = (det->params()[2]+det->params()[1])/2.;

    GeometricDet::GeometricDetContainer neg;
    GeometricDet::GeometricDetContainer pos;
    GeometricDet::GeometricDetContainer extneg;
    GeometricDet::GeometricDetContainer intneg;
    GeometricDet::GeometricDetContainer extpos;
    GeometricDet::GeometricDetContainer intpos;
    neg.clear();
    pos.clear();
    extneg.clear();
    intneg.clear();
    extpos.clear();
    intpos.clear();

    for(size_t i = 0; i< comp.size(); ++i) {
      auto component = det->component(i);
      if(component->translation().z()<0.){
	neg.emplace_back(component);
      }else{
	pos.emplace_back(component);
      }
    }

    for(auto & i : neg){
      double rPos = i->translation().Rho();
      if(rPos > layerRadius ){ 
	extneg.emplace_back(i);
      }else{
	intneg.emplace_back(i);
      }
    }

    for(auto & po : pos){
      double rPos = po->translation().Rho();
      if(rPos > layerRadius ){ 
	extpos.emplace_back(po);
      }else{
	intpos.emplace_back(po);
      }
    }

    trackerStablePhiSort(extneg.begin(), extneg.end(), getPhi);
    trackerStablePhiSort(extpos.begin(), extpos.end(), getPhi);
    trackerStablePhiSort(intneg.begin(), intneg.end(), getPhi);
    trackerStablePhiSort(intpos.begin(), intpos.end(), getPhi);

    for(uint32_t i=0;i<intneg.size();i++){
      uint32_t temp=i+1;
      temp|=(1<<8); // 1 : SubDetector Side TIB-
      temp|=(1<<6); // 1 : Layer Part int
      intneg[i]->setGeographicalID(DetId(temp));
    }

    for(uint32_t i=0;i<extneg.size();i++){
      uint32_t temp=i+1;
      temp|=(1<<8); // 1 : SubDetector Side TIB-
      temp|=(2<<6); // 2 : Layer Part ext
      extneg[i]->setGeographicalID(DetId(temp));
    }

    for(uint32_t i=0;i<intpos.size();i++){
      uint32_t temp=i+1;
      temp|=(2<<8); // 2 : SubDetector Side TIB+
      temp|=(1<<6); // 1 : Layer Part int 
      intpos[i]->setGeographicalID(DetId(temp));
    }

    for(uint32_t i=0;i<extpos.size();i++){
      uint32_t temp=i+1;
      temp|=(2<<8); // 2 : SubDetector Side TIB+
      temp|=(2<<6); // 2 : Layer Part ext 
      extpos[i]->setGeographicalID(DetId(temp));
    }
    
    
    det->clearComponents();
    det->addComponents(intneg);
    det->addComponents(extneg);
    det->addComponents(intpos);
    det->addComponents(extpos);
    
  }else if(comp.front()->type()== GeometricDet::rod){
    GeometricDet::GeometricDetContainer neg;
    GeometricDet::GeometricDetContainer pos;
    neg.clear();
    pos.clear();
    
    for(size_t i=0; i<comp.size(); ++i) {
      auto component = det->component(i);
      if(component->translation().z()<0.){
	neg.emplace_back(component);
      }else{
	pos.emplace_back(component);
      }
    }

    trackerStablePhiSort(neg.begin(), neg.end(), getPhi);
    trackerStablePhiSort(pos.begin(), pos.end(), getPhi);
    
    for(uint32_t i=0; i<neg.size();i++){      
      uint32_t temp = i+1;
      temp|=(1<<7);
      neg[i]->setGeographicalID(DetId(temp));
    }
    
    for(uint32_t i=0; i<pos.size();i++){
      uint32_t temp = i+1;
      temp|=(2<<7);
      pos[i]->setGeographicalID(DetId(temp));
    }
    
    det->clearComponents();
    det->addComponents(neg);
    det->addComponents(pos);
    
  }else if(det->components().front()->type()== GeometricDet::ladder){

    trackerStablePhiSort(comp.begin(), comp.end(), getPhi);

    for(uint32_t i=0; i<comp.size();i++){
      det->component(i)->setGeographicalID(DetId(i+1));
    }    

  }else{
    edm::LogError("CmsTrackerLayerBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type();
  }

}

