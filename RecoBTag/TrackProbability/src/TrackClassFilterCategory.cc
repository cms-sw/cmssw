#include "RecoBTag/XMLCalibration/interface/CalibrationCategory.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationXML.h"

#include "RecoBTag/TrackProbability/interface/TrackClassFilter.h"
#include "RecoBTag/TrackProbability/interface/TrackClassFilterCategory.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>

//#include <iostream>
//#include <stdlib.h>




bool TrackClassFilterCategory::match(const TrackClassFilterInput &input) const 
{
   return filter.apply(input.track, input.jet, input.vertex);
}

void TrackClassFilterCategory::readFromDOM( XERCES_CPP_NAMESPACE::DOMElement * dom)
 {
   float pmin = CalibrationXML::readAttribute<float>(dom,"pmin");
   float pmax = CalibrationXML::readAttribute<float>(dom,"pmax");
   float etamin = CalibrationXML::readAttribute<float>(dom,"etamin");
   float etamax = CalibrationXML::readAttribute<float>(dom,"etamax");
   int nhmin = CalibrationXML::readAttribute<int>(dom,"numhitsmin");
   int nhmax = CalibrationXML::readAttribute<int>(dom,"numhitsmax");
   int npmin = CalibrationXML::readAttribute<int>(dom,"numpixelmin");
   int npmax = CalibrationXML::readAttribute<int>(dom,"numpixelmax");
   float chimin = CalibrationXML::readAttribute<float>(dom,"chimin");
   float chimax = CalibrationXML::readAttribute<float>(dom,"chimax");
   int withFirst = CalibrationXML::readAttribute<int>(dom,"withfirstpixelhit");

   filter.set( pmin, pmax, etamin, etamax, nhmin, nhmax,npmin, npmax,chimin,chimax,withFirst) ;

 }
 
void TrackClassFilterCategory::saveToDOM( XERCES_CPP_NAMESPACE::DOMElement * dom)
{
   CalibrationXML::writeAttribute(dom,"pmin",filter.pMin);
   CalibrationXML::writeAttribute(dom,"pmax",filter.pMax);
   CalibrationXML::writeAttribute(dom,"etamin",filter.etaMin);
   CalibrationXML::writeAttribute(dom,"etamax",filter.etaMax);
   CalibrationXML::writeAttribute(dom,"numhitsmin",filter.nHitsMin);
   CalibrationXML::writeAttribute(dom,"numhitsmax",filter.nHitsMax);
   CalibrationXML::writeAttribute(dom,"numpixelmin",filter.nPixelHitsMin);
   CalibrationXML::writeAttribute(dom,"numpixelmax",filter.nPixelHitsMax);
   CalibrationXML::writeAttribute(dom,"chimin",filter.chiMin);
   CalibrationXML::writeAttribute(dom,"chimax",filter.chiMax);
   CalibrationXML::writeAttribute(dom,"withfirstpixelhit",filter.withFirstPixel);
}

