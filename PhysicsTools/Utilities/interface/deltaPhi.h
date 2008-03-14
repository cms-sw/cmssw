#ifndef PhysicsTools_Utilities_deltaPhi_h
#define PhysicsTools_Utilities_deltaPhi_h
/* function to compute deltaPhi
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 */
#include <cmath>

namespace reco {
  template <class T> 
  T deltaPhi (T phi1, T phi2) { 
    T result = phi1 - phi2;
    while (result > M_PI) result -= 2*M_PI;
    while (result <= -M_PI) result += 2*M_PI;
    return result;
  }

}

#endif
