#ifndef IsolationUtils_PtIsolationAlgo_h
#define IsolationUtils_PtIsolationAlgo_h
/* \class PtIsolationAlgo<T, C>
 *
 * \author Francesco Fabozzi, INFN
 */
#include "DataFormats/Math/interface/deltaR.h"

template <typename T, typename C>
class PtIsolationAlgo {
public:
  typedef double value_type;
  PtIsolationAlgo() { }
  PtIsolationAlgo( double dRMin, double dRMax, double dzMax,
		   double d0Max, double ptMin ) :
    dRMin_( dRMin ), dRMax_( dRMax ), dzMax_( dzMax ),
    d0Max_( d0Max ), ptMin_( ptMin ) { }
  double operator()(const T &, const C &) const;

private:
  double dRMin_, dRMax_, dzMax_, d0Max_, ptMin_;
};

template <typename T, typename C>
double PtIsolationAlgo<T, C>::operator()(const T & cand, const C & elements) const {
  double ptSum = 0;
  double candVz = cand.vz();
  double candEta = cand.eta();
  double candPhi = cand.phi();
  for( typename C::const_iterator elem = elements.begin(); elem != elements.end(); ++ elem ) {
    double elemPt = elem->pt();
    if ( elemPt < ptMin_ ) continue;
    double elemVx = elem->vx();
    double elemVy = elem->vy();
    double elemD0 = sqrt( elemVx * elemVx + elemVy * elemVy );
    if ( elemD0 > d0Max_ ) continue;
    double dz = fabs( elem->vz() - candVz );
    if ( dz > dzMax_ ) continue;
    double dR = deltaR( elem->eta(), elem->phi(), candEta, candPhi );
    if ( (dR > dRMax_) || (dR < dRMin_) ) continue;
    ptSum += elemPt;
  }
  return ptSum;
}

#endif
