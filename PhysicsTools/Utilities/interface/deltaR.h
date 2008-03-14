#ifndef PhysicsTools_Utilities_deltaR_h
#define PhysicsTools_Utilities_deltaR_h
/* functions to compute deltaR
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 */
#include "PhysicsTools/Utilities/interface/deltaPhi.h"
#include <cmath>

namespace reco {

  template <class T1, class T2>
  T1 deltaR2 (T1 eta1, T2 phi1, T1 eta2, T2 phi2) {
    T1 deta = eta1 - eta2;
    T2 dphi = deltaPhi(phi1, phi2);
    return deta*deta + dphi*dphi;
  }
  
  template <class T1, class T2>
  T1 deltaR (T1 eta1, T2 phi1, T1 eta2, T2 phi2) {
    return sqrt(deltaR2 (eta1, phi1, eta2, phi2));
  }
  
  template<typename T1, typename T2>
  double deltaR2(const T1 & t1, const T2 & t2) {
    return deltaR2(t1.eta(), t1.phi(), t2.eta(), t2.phi());
  } 
  
  template<typename T1, typename T2>
  double deltaR(const T1 & t1, const T2 & t2) {
    return deltaR(t1.eta(), t1.phi(), t2.eta(), t2.phi());
  } 

}

#endif
