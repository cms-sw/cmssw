#ifndef IsolationUtils_CalIsolationAlgoNoExp_h
#define IsolationUtils_CalIsolationAlgoNoExp_h
/* \class CalIsolationAlgoNoExp<T1, C2>
 *
 * \author Christian Autermann, U Hamburg
 */
#include "PhysicsTools/Utilities/interface/Math.h"

template <typename T1, typename C2>
class CalIsolationAlgoNoExp {
public:
  typedef double value_type;
  CalIsolationAlgoNoExp( );
  CalIsolationAlgoNoExp( double dRMin, double dRMax) : dRMin_( dRMin ), dRMax_( dRMax ) { }
  ~CalIsolationAlgoNoExp() { } 
  double operator()(const T1 &, const C2 &) const;

private:
  double dRMin_, dRMax_;
};

template <typename T1, typename C2> double CalIsolationAlgoNoExp<T1,C2>::
operator()(const T1 & cand, const C2 & elements) const {
  double etSum = 0;
  for( typename C2::const_iterator elem = elements.begin(); 
       elem != elements.end(); ++elem ) {
    double dR = deltaR( elem->eta(), elem->phi(), 
                        cand.eta(), cand.phi() );
    if ( dR < dRMax_ && dR > dRMin_ ) {
      etSum += elem->et();
    }
  }
  return etSum;
}

#endif
