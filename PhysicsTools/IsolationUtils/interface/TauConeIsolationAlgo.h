#ifndef PhysicsTools_IsolationUtils_TauConeIsolationAlgo_h
#define PhysicsTools_IsolationUtils_TauConeIsolationAlgo_h

#include "DataFormats/Common/interface/RefVector.h"

#include "PhysicsTools/Utilities/interface/Math.h"

template <typename T, typename C, typename M>
  class TauConeIsolationAlgo {
   public:
    TauConeIsolationAlgo() {}
    ~TauConeIsolationAlgo() {}
  
    const edm::RefVector<C> operator()(const T & coneAxis, double coneSize, const edm::RefVector<C> & elements, const M & metric) const;
};

template <typename T, typename C, typename M>
  const edm::RefVector<C> TauConeIsolationAlgo<T, C, M>::operator()(const T & coneAxis, double coneSize, const edm::RefVector<C> & elements, const M & metric) const 
{
  edm::RefVector<C> elementsInCone;
  for( typename edm::RefVector<C>::const_iterator element = elements.begin(); 
       element != elements.end(); ++element ) {
    double distance = metric(coneAxis, (*element)->momentum());

    if ( distance <= coneSize ) {
      elementsInCone.push_back(*element);
    }
  }

  return elementsInCone;
}

#endif
