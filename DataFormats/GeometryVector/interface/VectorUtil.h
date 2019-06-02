#ifndef GeometryVector_Geom_Util_h
#define GeometryVector_Geom_Util_h

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <cmath>

namespace Geom {
  using reco::deltaPhi;
  using reco::deltaR;
  using reco::deltaR2;

  /** Definition of ordering of azimuthal angles.
   *  phi1 is less than phi2 if the angle covered by a point going from
   *  phi1 to phi2 in the counterclockwise direction is smaller than pi.
   *  It makes sense only if ALL phis are in a single hemisphere...
   */
  inline bool phiLess(float phi1, float phi2) { return deltaPhi(phi1, phi2) < 0; }
  inline bool phiLess(double phi1, double phi2) { return deltaPhi(phi1, phi2) < 0; }
  template <class Vector1, class Vector2>
  bool phiLess(const Vector1& v1, const Vector2& v2) {
    return deltaPhi(v1.phi(), v2.phi()) < 0.;
  }

}  // namespace Geom

#endif
