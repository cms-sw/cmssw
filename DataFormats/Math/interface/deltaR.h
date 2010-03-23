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

  inline double deltaR2(double eta1, double phi1, double eta2, double phi2) {
    double deta = eta1 - eta2;
    double dphi = deltaPhi(phi1, phi2);
    return deta*deta + dphi*dphi;
  }
  
  inline double deltaR(double eta1, double phi1, double eta2, double phi2) {
    return std::sqrt(deltaR2 (eta1, phi1, eta2, phi2));
  }
  
  template<typename T1, typename T2>
  inline double deltaR2(const T1 & t1, const T2 & t2) {
    return deltaR2(t1.eta(), t1.phi(), t2.eta(), t2.phi());
  } 
  
  template<typename T1, typename T2>
  inline double deltaR(const T1 & t1, const T2 & t2) {
    return deltaR(t1.eta(), t1.phi(), t2.eta(), t2.phi());
  } 

  template <class T>
  T deltaR2 (T eta1, T phi1, T eta2, T phi2) {
    T deta = eta1 - eta2;
    T dphi = deltaPhi (phi1, phi2);
    return deta*deta + dphi*dphi;
  }

  template <class T>
    T deltaR (T eta1, T phi1, T eta2, T phi2) {
    return std::sqrt (deltaR2 (eta1, phi1, eta2, phi2));
  }

}

using reco::deltaR2;
using reco::deltaR;

template<typename T1, typename T2 = T1>
struct DeltaR {
  double operator()( const T1 & t1, const T2 & t2 ) const {
    return reco::deltaR(t1, t2);
  }

};
#endif
