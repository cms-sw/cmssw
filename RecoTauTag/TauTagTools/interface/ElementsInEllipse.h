#ifndef ElementsInEllipse_h
#define ElementsInEllipse_h

#include "RecoTauTag/TauTagTools/interface/Ellipse.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include <utility>

template  <typename T, typename C>
  class ElementsInEllipse{
    public:
      ElementsInEllipse(){}
      ~ElementsInEllipse(){}
     
      const std::pair<std::vector<edm::Ptr<C> >, std::vector<edm::Ptr<C> > > operator()(const T& axis, double rPhi, double rEta, const std::vector<edm::Ptr<C> >& elements)const{
	std::vector<edm::Ptr<C> > elementsInEllipse;
	std::vector<edm::Ptr<C> > elementsOutEllipse;
	for(typename std::vector<edm::Ptr<C> >::const_iterator element = elements.begin(); element != elements.end(); ++element){
	  double distance = ellipse(axis, (*element)->momentum(), rPhi, rEta);
	  if(distance <= 1.)elementsInEllipse.push_back(*element);
	  else elementsOutEllipse.push_back(*element);
	}
        std::pair<std::vector<edm::Ptr<C> >, std::vector<edm::Ptr<C> > > theInOutPair(elementsInEllipse, elementsOutEllipse);
	return theInOutPair;
      }
  };
#endif
