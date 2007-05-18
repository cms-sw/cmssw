#ifndef IsolationUtils_PtIsolationAlgo_h
#define IsolationUtils_PtIsolationAlgo_h
/* \class PtIsolationAlgo<T, C>
 *
 * \author Francesco Fabozzi, INFN
 */
#include "PhysicsTools/Utilities/interface/Math.h"

template <typename T, typename C>
class PtIsolationAlgo {
public:
  typedef double value_type;
  PtIsolationAlgo() { }
  PtIsolationAlgo( double dRMin, double dRMax, double dzMax ) :
    dRMin_( dRMin ), dRMax_( dRMax ), dzMax_( dzMax ) { }
  double operator()(const T &, const C &) const;

private:
  double dRMin_, dRMax_, dzMax_;
};

template <typename T, typename C>
double PtIsolationAlgo<T, C>::operator()(const T & cand, const C & elements) const {
  double ptSum = 0;
  double candVz = cand.vz();
  double candEta = cand.eta();
  double candPhi = cand.phi();
  for( typename C::const_iterator elem = elements.begin(); elem != elements.end(); ++ elem ) {
    double dz = fabs( elem->vz() - candVz );
    double dR = deltaR( elem->eta(), elem->phi(), candEta, candPhi );
    if ( dz < dzMax_ &&  dR < dRMax_ && dR > dRMin_ ) {
      ptSum += elem->pt();
    }
  }
  return ptSum;
}

#endif
