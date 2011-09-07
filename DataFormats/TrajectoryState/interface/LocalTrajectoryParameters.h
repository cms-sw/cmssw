#ifndef _TRACKER_LOCALTRAJECTORYPARAMETERS_H_
#define _TRACKER_LOCALTRAJECTORYPARAMETERS_H_

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include "DataFormats/TrajectoryState/interface/TrackCharge.h"

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
  LocalTrajectoryParameters(const AlgebraicVector& v, double aPzSign, bool charged = true) {
    theQbp  = v[0];
    theDxdz = v[1];
    theDydz = v[2];
    theX    = v[3];
    theY    = v[4];
    thePzSign = aPzSign;
    if ( charged )
      theCharge = theQbp>0 ? 1 : -1;
    else
      theCharge = 0;
  }

  /** Constructor from vector of parameters.
   *
   *  Expects a vector of parameters as defined above, plus the sign of p_z.
   *  For charged particles the charge will be determined by the sign of
   *  the first element. For neutral particles the last argument should be false, 
   *  in which case the charge of the first element will be neglected.
   */
  LocalTrajectoryParameters(const AlgebraicVector5& v, double aPzSign, bool charged = true) {
    theQbp  = v[0];
    theDxdz = v[1];
    theDydz = v[2];
    theX    = v[3];
    theY    = v[4];
    thePzSign = aPzSign;
    if ( charged )
      theCharge = theQbp>0 ? 1 : -1;
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
  LocalTrajectoryParameters(double aQbp, double aDxdz, double aDydz,
                            double aX, double aY, double aPzSign, bool charged = true) :
    theDxdz(aDxdz), theDydz(aDydz),
    theX(aX), theY(aY), thePzSign(aPzSign) {
    if ( charged ) {
      theQbp = aQbp;
      theCharge = theQbp>0 ? 1 : -1;
    }
    else {
      theQbp = aQbp;
      theCharge = 0;
    }
  }

  /// Constructor from local position, momentum and charge.
  LocalTrajectoryParameters( const LocalPoint& pos, const LocalVector& p,
			     TrackCharge charge) :
    theQbp( charge/p.mag()), theDxdz( p.x()/p.z()), theDydz( p.y()/p.z()), 
    theX( pos.x()), theY(pos.y()), thePzSign( p.z()>0. ? 1.:-1.), theCharge(charge) {
    if ( charge==0 )  theQbp = 1./p.mag();
  }

// access

  /// Local x and y position coordinates.
  LocalPoint position() const {
    return LocalPoint(theX, theY);
  }

  /// Momentum vector in the local frame. 
  LocalVector momentum() const {
    double p = 1./fabs(theQbp);
    if ( p>1.e9 )  p = 1.e9;
    double dz = thePzSign/sqrt(1. + theDxdz*theDxdz + theDydz*theDydz);
    double dx = dz*theDxdz;
    double dy = dz*theDydz;
    return LocalVector(dx*p, dy*p, dz*p);
  }

  /// Charge (-1, 0 or 1)
  TrackCharge charge() const {return theCharge;}

  /// Signed inverse momentum q/p (zero for neutrals).
  double signedInverseMomentum() const {
    return charge()==0 ? 0. : theQbp;
  }

  /** Vector of parameters with signed inverse momentum.
   *
   *  Vector of parameters as defined above, with the
   *  first element = q/p .
   */
  AlgebraicVector vector_old() const {
    AlgebraicVector v(5);
    v[0] = signedInverseMomentum();
    v[1] = theDxdz;
    v[2] = theDydz;
    v[3] = theX;
    v[4] = theY;
    return v;
  }

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
    v[0] = theQbp;    // signed in case of charged particles, 1/p for neutrals
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
  AlgebraicVector mixedFormatVector_old() const {
    AlgebraicVector v(5);
    v[0] = theQbp;    // signed in case of charged particles, 1/p for neutrals
    v[1] = theDxdz;
    v[2] = theDydz;
    v[3] = theX;
    v[4] = theY;
    return v;
  }

  /// Sign of the z-component of the momentum in the local frame.
  double pzSign() const {
    return thePzSign;
  }

  /// Update of momentum by a scalar dP.
  bool updateP(double dP) {
    double p = 1./fabs(theQbp);
    if ((p += dP) <= 0.) return false;
    double newQbp = theQbp > 0. ? 1./p : -1./p;
    theQbp = newQbp;
    return true;
  }


  double qbp() const { return theQbp;}
  double dxdz() const { return theDxdz;}
  double dydz() const { return theDydz;}


private:
  double theQbp;    ///< q/p (charged) or 1/p (neutral)
  double theDxdz;   ///< tangent of direction in local x vs. z
  double theDydz;   ///< tangent of direction in local y vs. z
  double theX;      ///< local x position
  double theY;      ///< local y position

  double thePzSign; ///< sign of local pz

  TrackCharge theCharge; ///< charge

};

#endif
