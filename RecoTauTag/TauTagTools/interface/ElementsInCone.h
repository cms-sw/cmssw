#ifndef ElementsInCone_h
#define ElementsInCone_h

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/angle.h"
#include "DataFormats/Math/interface/deltaR.h"

template <typename T, typename M, typename C>
  class ElementsInCone {
   public:
    ElementsInCone() {}
    ~ElementsInCone() {}  
    const edm::RefVector<C> operator()(const T& coneAxis,const M& coneMetric,double coneSize,const edm::RefVector<C>& elements)const;
};

template <typename T,typename M,typename C> const edm::RefVector<C> ElementsInCone<T, M, C>::operator()(const T& coneAxis,const M& coneMetric,double coneSize,const edm::RefVector<C>& elements)const{
  edm::RefVector<C> elementsInCone;
  for(typename edm::RefVector<C>::const_iterator element=elements.begin();element!=elements.end();++element) {
    double distance = coneMetric(coneAxis,(*element)->momentum());
    if (distance<=coneSize)elementsInCone.push_back(*element);
  }
  return elementsInCone;
}

#endif

