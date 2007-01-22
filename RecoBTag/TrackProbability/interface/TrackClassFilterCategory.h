#ifndef TRACKCLASSFILTERCATEGORY_H
#define TRACKCLASSFILTERCATEGORY_H

#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h"
#include "CondFormats/BTagObjects/interface/TrackProbabilityCategoryData.h"


#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>

#include <iostream>

class TrackClassFilterInput
{
public:
 TrackClassFilterInput(const reco::Track & t,const reco::Jet &j, const reco::Vertex & v) : 
                       track(t), jet(j), vertex(v) {}

 const reco::Track & track;
 const reco::Jet & jet;
 const reco::Vertex & vertex;
};



class TrackClassFilterCategory 
{
public:
  TrackClassFilterCategory(){}
 TrackClassFilterCategory(const TrackProbabilityCategoryData & d) : filter(d)  {}
//.pMin(), d.pMax(),
  //               d.etaMin(), d.etaMax(),d.nHitsMin(), d.nHitsMax(),
    //             d.nPixelHitsMin(),d.nPixelHitsMax()) 

 typedef TrackClassFilterInput Input;

 bool match(const TrackClassFilterInput &input) const;
  void readFromDOM( XERCES_CPP_NAMESPACE::DOMElement * dom);
 void saveToDOM( XERCES_CPP_NAMESPACE::DOMElement * dom);
 
 
 private:
  TrackClassFilter filter; 
};

#endif
