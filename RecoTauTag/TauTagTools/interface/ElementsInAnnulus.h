#ifndef ElementsInAnnulus_h
#define ElementsInAnnulus_h

#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Math/interface/angle.h"
#include "DataFormats/Math/interface/deltaR.h"

template <typename T, typename M, typename N, typename C>
  class ElementsInAnnulus {
   public:
    ElementsInAnnulus() {}
    ~ElementsInAnnulus() {}  
    const edm::RefVector<C> operator()(const T& coneAxis,const M& innerconeMetric,double innerconeSize,const N& outerconeMetric,double outerconeSize,const edm::RefVector<C>& elements)const;
};

template <typename T, typename M, typename N,typename C> const edm::RefVector<C> ElementsInAnnulus<T,M,N,C>::operator()(const T& coneAxis,const M& innerconeMetric,double innerconeSize,const N& outerconeMetric,double outerconeSize,const edm::RefVector<C> & elements)const{
  edm::RefVector<C> elementsInBand;
  for(typename edm::RefVector<C>::const_iterator element=elements.begin();element!=elements.end();++element) {
    double innerconeMetric_distance=innerconeMetric(coneAxis,(*element)->momentum());
    double outerconeMetric_distance=outerconeMetric(coneAxis,(*element)->momentum());
    if (innerconeMetric_distance>innerconeSize && outerconeMetric_distance<=outerconeSize)elementsInBand.push_back(*element);
  }
  return elementsInBand;
}

#endif

