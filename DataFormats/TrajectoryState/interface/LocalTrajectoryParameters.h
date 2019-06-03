#ifndef _TRACKER_LOCALTRAJECTORYPARAMETERS_H_
#define _TRACKER_LOCALTRAJECTORYPARAMETERS_H_

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

#include <cmath>

/** Class providing access to a set of relevant parameters of a trajectory
 *  in a local, Cartesian frame. The set consists of the following parameters: <BR> <BR>
 *  
 *  q/p : charged particles: charge (plus or minus one) divided by magnitude of momentum <BR>
 *        neutral particles: inverse magnitude of momentum <BR>
 *  dxdz : direction tangent in local xz-plane <BR>
 *  dydz : direction tangent in local yz-plane <BR>
 *  x : local x-coordinate <BR>
 *  y : local y-coordinate <BR> <BR>
 *
 *  In addition, the sign of local p_z is needed to fully define the direction of the track
 *  in this local frame.
 */

class LocalTrajectoryParameters {
public:
  // construct

  LocalTrajectoryParameters() {}

  /** Constructor from vector of parameters.
   *
   *  Expects a vector of parameters as defined above, plus the sign of p_z.
   *  For charged particles the charge will be determined by the sign of
   *  the first element. For neutral particles the last argument should be false, 
   *  in which case the charge of the first element will be neglected.
   */
  LocalTrajectoryParameters(const AlgebraicVector5& v, float aPzSign, bool charged = true) {
    theQbp = v[0];
    theDxdz = v[1];
    theDydz = v[2];
    theX = v[3];
    theY = v[4];
    thePzSign = aPzSign;
    if (charged)
      theCharge = theQbp > 0 ? 1 : -1;
    else
      theCharge = 0;
  }

  /** Constructor from individual parameters.
   *
   *  Expects parameters as defined above, plus the sign of p_z.
   *  For charged particles the charge will be determined by the sign of
   *  the first argument. For neutral particles the last argument should be false, 
   *  in which case the charge of the first argument will be neglected.
   */
  LocalTrajectoryParameters(float aQbp, float aDxdz, float aDydz, float aX, float aY, float aPzSign, bool charged = true)
      : theDxdz(aDxdz), theDydz(aDydz), theX(aX), theY(aY), thePzSign(aPzSign > 0 ? 1 : -1) {
    if (charged) {
      theQbp = aQbp;
      theCharge = theQbp > 0 ? 1 : -1;
    } else {
      theQbp = aQbp;
      theCharge = 0;
    }
  }

  /// Constructor from local position, momentum and charge.
  LocalTrajectoryParameters(const LocalPoint& pos, const LocalVector& p, TrackCharge charge)
      : theQbp(charge / p.mag()),
        theDxdz(p.x() / p.z()),
        theDydz(p.y() / p.z()),
        theX(pos.x()),
        theY(pos.y()),
        thePzSign(p.z() > 0. ? 1 : -1),
        theCharge(charge) {
    if (charge == 0)
      theQbp = 1.f / p.mag();
  }

  // access

  /// Local x and y position coordinates.
  LocalPoint position() const { return LocalPoint(theX, theY); }

  /// Momentum vector in the local frame.
  LocalVector momentum() const {
    float op = std::abs(theQbp);
    if (op < 1.e-9f)
      op = 1.e-9f;
    float pz = float(thePzSign) / (op * std::sqrt(1.f + theDxdz * theDxdz + theDydz * theDydz));
    float px = pz * theDxdz;
    float py = pz * theDydz;
    return LocalVector(px, py, pz);
  }

  /// Momentum vector unit in the local frame.
  LocalVector direction() const {
    float dz = float(thePzSign) / std::sqrt(1.f + theDxdz * theDxdz + theDydz * theDydz);
    float dx = dz * theDxdz;
    float dy = dz * theDydz;
    return LocalVector(dx, dy, dz);
  }

  /// Momentum vector unit in the local frame.
  LocalVector directionNotNormalized() const { return LocalVector(theDxdz, theDydz, 1.f); }

  /// Charge (-1, 0 or 1)
  TrackCharge charge() const { return theCharge; }

  /// Signed inverse momentum q/p (zero for neutrals).
  float signedInverseMomentum() const { return charge() == 0 ? 0.f : theQbp; }

  /** Vector of parameters with signed inverse momentum.
   *
   *  Vector of parameters as defined above, with the
   *  first element = q/p .
   */
  AlgebraicVector5 vector() const {
    AlgebraicVector5 v;
    v[0] = signedInverseMomentum();
    v[1] = theDxdz;
    v[2] = theDydz;
    v[3] = theX;
    v[4] = theY;
    return v;
  }

  /** Vector of parameters in internal representation.
   *
   *  Vector of parameters as defined above, with the first
   *  element = q/p for charged and = 1/p for neutral.
   */
  AlgebraicVector5 mixedFormatVector() const {
    AlgebraicVector5 v;
    v[0] = theQbp;  // signed in case of charged particles, 1/p for neutrals
    v[1] = theDxdz;
    v[2] = theDydz;
    v[3] = theX;
    v[4] = theY;
    return v;
  }

  /// Sign of the z-component of the momentum in the local frame.
  float pzSign() const { return thePzSign; }

  /// Update of momentum by a scalar dP.
  bool updateP(float dP) {
    float p = 1.f / std::abs(theQbp);
    if ((p += dP) <= 0.f)
      return false;
    float newQbp = theQbp > 0. ? 1.f / p : -1.f / p;
    theQbp = newQbp;
    return true;
  }

  float qbp() const { return theQbp; }
  float dxdz() const { return theDxdz; }
  float dydz() const { return theDydz; }
  float absdz() const { return 1.f / std::sqrt(1.f + theDxdz * theDxdz + theDydz * theDydz); }

private:
  float theQbp;   ///< q/p (charged) or 1/p (neutral)
  float theDxdz;  ///< tangent of direction in local x vs. z
  float theDydz;  ///< tangent of direction in local y vs. z
  float theX;     ///< local x position
  float theY;     ///< local y position

  short thePzSign;  ///< sign of local pz

  short theCharge;  ///< charge
};

#endif
