#ifndef Utilities_DeltaR_h
#define Utilities_DeltaR_h
/* \class DeltaR
 *
 * returns DeltaR between two objects
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include <Math/VectorUtil.h>

template<typename T>
struct DeltaR {
  double operator()( const T & t1, const T & t2 ) const {
    return ROOT::Math::VectorUtil::DeltaR( t1.p4(), t2.p4() );
  }
};

#endif
