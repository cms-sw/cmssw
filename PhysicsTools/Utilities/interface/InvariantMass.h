#ifndef RecoAlgos_InvariantMass_h
#define RecoAlgos_InvariantMass_h
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

template<typename T>
struct InvariantMass {
  InvariantMass() { }
  explicit InvariantMass( const edm::ParameterSet & ) { }
  double operator()( const T & t1, const T & t2 ) const {
    return ( t1.momentum() + t2.momentum() ).mass();
  }
};

#endif
