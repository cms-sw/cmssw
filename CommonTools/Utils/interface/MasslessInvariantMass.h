#ifndef CommonTools_Utils_MasslessInvariantMass_h
#define CommonTools_Utils_MasslessInvariantMass_h
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

struct MasslessInvariantMass {
  template<typename T1, typename T2>
  double operator()( const T1 & t1, const T2 & t2 ) const {
    math::XYZVector p1 = t1.momentum(), p2 = t2.momentum();
    math::XYZTLorentzVector v1( p1.x(), p1.y(), p1.z(), p1.r() ), v2( p2.x(), p2.y(), p2.z(), p2.r() );
    return ( v1 + v2 ).mass();
  }
};

#endif
