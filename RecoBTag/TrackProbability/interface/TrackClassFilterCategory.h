#ifndef TRACKCLASSFILTERCATEGORY_H
#define TRACKCLASSFILTERCATEGORY_H

#include "RecoBTag/XMLCalibration/interface/CalibrationCategory.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>

class TrackClassFilterInput
{
public:
 TrackClassFilterInput(const reco::Track & t,const reco::Jet &j, const reco::Vertex & v) : 
                       track(t), jet(j), vertex(v) {}

 const reco::Track & track;
 const reco::Jet & jet;
 const reco::Vertex & vertex;
};


class TrackClassFilterCategory : public CalibrationCategory<TrackClassFilterInput>
{
public:
 TrackClassFilterCategory() :  CalibrationCategory<TrackClassFilterInput>() {}
 
 virtual string name()	{return "TrackClassFilterCategory";} 
 
 bool match(const TrackClassFilterInput &input) const;
 void readFromDOM( XERCES_CPP_NAMESPACE::DOMElement * dom);
 void saveToDOM( XERCES_CPP_NAMESPACE::DOMElement * dom);
 
 void dump() 
 {
 cout << " pMin " << filter.pMin() << 
	 " pMax " <<filter.pMax() <<  endl<<
	 " etaMin " <<filter.etaMin() << 
	 " etaMax " <<filter.etaMax() <<  endl<<
	 " nHitMin " <<filter.nHitMin() <<
	 " nHitMax " <<filter.nHitMax() <<  endl <<
	 " nPixelMin " <<filter.nPixelMin() << 
	 " nPixelMax " <<filter.nPixelMax() << endl;
 }
 
 private:
  TrackClassFilter filter; 
};

#endif
