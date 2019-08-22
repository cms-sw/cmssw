#ifndef Math_notmalizedPhi_h
#define Math_notmalizedPhi_h
#include "DataFormats/Math/interface/deltaPhi.h"
#include <algorithm>

// return a value of phi into interval [-pi,+pi]
template <typename T>
constexpr T normalizedPhi(T phi) {
  return reco::reduceRange(phi);
}

// cernlib V306
template <typename T>
constexpr T proxim(T b, T a) {
  constexpr T c1 = 2. * M_PI;
  constexpr T c2 = 1 / c1;
  return b + c1 * std::round(c2 * (a - b));
}

#include <iostream>

// smallest range
template <typename T>
constexpr bool checkPhiInSymRange(T phi, T phi1, T phi2, float maxDphi = float(M_PI)) {
  // symmetrize
  if (phi2 < phi1)
    std::swap(phi1, phi2);
  return checkPhiInRange(phi, phi1, phi2, maxDphi);
}

// counterclock-wise range
template <typename T>
constexpr bool checkPhiInRange(T phi, T phi1, T phi2, float maxDphi = float(M_PI)) {
  phi2 = proxim(phi2, phi1);
  constexpr float c1 = 2. * M_PI;
  if (phi2 < phi1)
    phi2 += c1;
  auto dphi = std::min(maxDphi, 0.5f * (phi2 - phi1));
  auto phiA = phi1 + dphi;
  phi = proxim(phi, phiA);
  return std::abs(phiA - phi) < dphi;

  /* old "alternative algo"
    constexpr T c1 = 2.*M_PI;
    phi1 = normalizedPhi(phi1);
    phi2 = proxim(phi2,phi1);
    if (phi2<phi1) phi2+=c1;
    // phi & phi1 are in [-pi,pi] range...
    return ( (phi1 <= phi) & (phi <= phi2) )
//           || ( (phi1 <= phi-c1) & (phi-c1 <= phi2) )
           || ( (phi1 <= phi+c1) & (phi+c1 <= phi2) );
    */
}

#endif
