#ifndef RecoAlgos_InvariantMass_h
#define RecoAlgos_InvariantMass_h
#include "DataFormats/Math/interface/LorentzVector.h"

template<typename T>
struct InvariantMass {
  double operator()( const T & t1, const T & t2 ) const {
    return ( t1.momentum() + t2.momentum() ).mass();
  }
};

#endif
