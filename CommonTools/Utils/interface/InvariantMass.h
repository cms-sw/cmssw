#ifndef CommonTools_Utils_InvariantMass_h
#define CommonTools_Utils_InvariantMass_h
#include "DataFormats/Math/interface/LorentzVector.h"

template<typename T1, typename T2 = T1>
struct InvariantMass {
  double operator()( const T1 & t1, const T2 & t2 ) const {
    return ( t1.momentum() + t2.momentum() ).mass();
  }
};

#endif
