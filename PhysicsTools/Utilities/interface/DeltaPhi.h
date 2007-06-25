#ifndef Utilities_DeltaPhi_h
#define Utilities_DeltaPhi_h
/* \class DeltaPhi
 *
 * returns DeltaPhi between two objects
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Math.h"

template<typename T1, typename T2 = T1>
struct DeltaPhi {
  double operator()( const T1 & t1, const T2 & t2 ) const {
    return deltaPhi( t1, t2 );
  }
};

#endif
