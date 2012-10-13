#ifndef DataFormats_Math_deltaR_h
#define DataFormats_Math_deltaR_h
/* functions to compute deltaR
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 */
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

namespace reco {


  // assumption is that eta and phi are cached AND phi is computed using std::atan2
  // type is the type of T1::phi();
  template<typename T1, typename T2>
  inline auto  __attribute__((always_inline)) deltaR2(const T1 & t1, const T2 & t2) -> decltype(t1.phi()) {
    typedef  decltype(t1.phi()) Float;
    Float p1 = t1.phi(); 
    Float p2 = t2.phi(); 
    Float e1 = t1.eta(); 
    Float e2 = t2.eta(); 
    auto dp=std::abs(p1-p2); if (dp>Float(M_PI)) dp-=Float(2*M_PI);  
    return (e1-e2)*(e1-e2) + dp*dp;
  } 
  
  // do not use it: always cut in deltaR2!
  template<typename T1, typename T2>
  inline auto deltaR(const T1 & t1, const T2 & t2) -> decltype(t1.phi()) {
    return std::sqrt(deltaR2(t1,t2));
  } 


  //prefer the above...
  template <class T>
  T deltaR2 (T eta1, T phi1, T eta2, T phi2) {
    T deta = eta1 - eta2;
    T dphi = deltaPhi (phi1, phi2);
    return deta*deta + dphi*dphi;
  }

  // to be avoided
  template <class T>
  T deltaR (T eta1, T phi1, T eta2, T phi2) {
    return std::sqrt (deltaR2 (eta1, phi1, eta2, phi2));
  }

}

// woderful!  VI
using reco::deltaR2;
using reco::deltaR;

// obsolete use lambdas (and cut in deltaR2!)
template<typename T1, typename T2 = T1>
struct DeltaR {
  double operator()( const T1 & t1, const T2 & t2 ) const {
    return reco::deltaR(t1, t2);
  }

};
#endif
