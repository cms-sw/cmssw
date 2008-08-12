#ifndef PhysicsTools_Utilities_DeltaR_h
#define PhysicsTools_Utilities_DeltaR_h

/* \class DeltaR
 *
 * returns DeltaR between two objects
 *
 * \author Luca Lista, INFN
 */

#include "DataFormats/Math/interface/LorentzVector.h"
#include "PhysicsTools/Utilities/interface/deltaR.h"

template<typename T1, typename T2 = T1>
struct DeltaR {
  double operator()( const T1 & t1, const T2 & t2 ) const {
    return reco::deltaR(t1, t2);
  }
};

#endif
