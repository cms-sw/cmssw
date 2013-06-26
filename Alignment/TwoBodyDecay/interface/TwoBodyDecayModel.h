#ifndef Alignment_TwoBodyDecay_TwoBodyDecayModel_h
#define Alignment_TwoBodyDecay_TwoBodyDecayModel_h


/** /class TwoBodyDecayModel
 *
 *  This class provides useful methods needed for the two-body decay model used by implementations
 *  of e.g. class TwoBodyDecayEstimator or TwoBodyDecayLinearizationPointFinder.
 *
 *  /author Edmund Widl
 */

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecay.h"


class TwoBodyDecayModel
{
 public:
  
  TwoBodyDecayModel( double mPrimary = 91.1876, double mSecondary = 0.105658 );
  ~TwoBodyDecayModel();

  /** Rotates a vector pointing in z-direction into the direction defined by p=(px,py,pz).
   */
  AlgebraicMatrix rotationMatrix( double px, double py, double pz );

  /** Jacobian for transformation from curvilinear to cartesian representation (needs the z-component of
   *  magnetic field in inverse GeV as input).
   */
  AlgebraicMatrix curvilinearToCartesianJacobian( double rho, double theta, double phi, double zMagField );

  /** Jacobian for transformation from curvilinear to cartesian representation (needs the z-component of
   *  magnetic field in inverse GeV as input).
   */
  AlgebraicMatrix curvilinearToCartesianJacobian( AlgebraicVector curv, double zMagField );

  /** Convert vector from curvilinear to cartesian coordinates (needs the z-component of
   *  magnetic field in inverse GeV as input).
   */
  AlgebraicVector convertCurvilinearToCartesian( AlgebraicVector curv, double zMagField );

  /** Momenta of the secondaries in cartesian repraesentation.
   */
  const std::pair< AlgebraicVector, AlgebraicVector > cartesianSecondaryMomenta( const AlgebraicVector & param );

  /** Momenta of the secondaries in cartesian repraesentation.
   */
  const std::pair< AlgebraicVector, AlgebraicVector > cartesianSecondaryMomenta( const TwoBodyDecay & tbd );

  /** Momenta of the secondaries in cartesian repraesentation.
   */
  const std::pair< AlgebraicVector, AlgebraicVector > cartesianSecondaryMomenta( const TwoBodyDecayParameters & tbdparam );

 private:

  double thePrimaryMass;
  double theSecondaryMass;

};


#endif
