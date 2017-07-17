#ifndef ElementsInEllipseRef_h
#define ElementsInEllipseRef_h

#include "RecoTauTag/TauTagTools/interface/Ellipse.h"
#include "DataFormats/Common/interface/RefVector.h"
#include <utility>

template  <typename T, typename C>
  class ElementsInEllipse{
    public:
      ElementsInEllipseRef(){}
      ~ElementsInEllipseRef(){}
     
      const std::pair<edm::RefVector<C>, edm::RefVector<C> > operator()(const T& axis, double rPhi, double rEta, const edm::RefVector<C>& elements)const{
	edm::RefVector<C> elementsInEllipse;
	edm::RefVector<C> elementsOutEllipse;
	for(typename edm::RefVector<C>::const_iterator element = elements.begin(); element != elements.end(); ++element){
	  double distance = ellipse(axis, (*element)->momentum(), rPhi, rEta);
	  if(distance <= 1.)elementsInEllipse.push_back(*element);
	  else elementsOutEllipse.push_back(*element);
	}
        std::pair<edm::RefVector<C>, edm::RefVector<C> > theInOutPair(elementsInEllipse, elementsOutEllipse);
	return theInOutPair;
      }
  };
#endif
