#ifndef Utilities_Math_h
#define Utilities_Math_h
/*
 * Ported from orignla code in RecoJets 
 * by Fedor Ratnikov, FNAL
 */
#include <cmath>

template <class T> 
T deltaPhi (T phi1, T phi2) { 
  T result = phi1 - phi2;
  while (result > M_PI) result -= 2*M_PI;
  while (result <= -M_PI) result += 2*M_PI;
  return result;
}

template <class T>
T deltaR2 (T eta1, T phi1, T eta2, T phi2) {
  T deta = eta1 - eta2;
  T dphi = deltaPhi (phi1, phi2);
  return deta*deta + dphi*dphi;
}

template <class T>
T deltaR (T eta1, T phi1, T eta2, T phi2) {
  return sqrt (deltaR2 (eta1, phi1, eta2, phi2));
}

template<typename T1, typename T2>
double deltaR2( const T1 & t1, const T2 & t2 ) {
  return deltaR2( t1.eta(), t1.phi(), t2.eta(), t2.phi() );
} 

template<typename T1, typename T2>
double deltaR( const T1 & t1, const T2 & t2 ) {
  return deltaR( t1.eta(), t1.phi(), t2.eta(), t2.phi() );
} 

#endif
