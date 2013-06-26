#ifndef Alignment_TwoBodyDecay_TwoBodyDecayDerivatives_h
#define Alignment_TwoBodyDecay_TwoBodyDecayDerivatives_h

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecay.h"

/** /class TwoBodyDecayDerivatives
 *
 *  This class provides the derivatives matrices need by the class TwoBodyDecayEstimator.
 *
 *  /author Edmund Widl
 */


class TwoBodyDecayDerivatives
{

public:

  enum { dimension = 6 };

  enum DerivativeParameterName { px = 1, py = 2, pz = 3, theta = 4, phi = 5, mass = 6 };

  TwoBodyDecayDerivatives( double mPrimary = 91.1876, double mSecondary = 0.105658 );
  ~TwoBodyDecayDerivatives();

  /**Derivatives of the lab frame momenta (in cartesian representation) of the secondaries
   * w.r.t. z=(px,py,pz,theta,phi,m).
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > derivatives( const TwoBodyDecay & tbd ) const;

  /**Derivatives of the lab frame momenta (in cartesian representation) of the secondaries
   * w.r.t. z=(px,py,pz,theta,phi,m).
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > derivatives( const TwoBodyDecayParameters & param ) const;

  /**Derivatives of the lab frame momenta (in cartesian representation) of the secondaries
   * w.r.t. the selected parameters.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > selectedDerivatives( const TwoBodyDecay & tbd,
									   const std::vector< bool > & selector ) const;

  /**Derivatives of the lab frame momenta (in cartesian representation) of the secondaries
   * w.r.t. the selected parameters.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > selectedDerivatives( const TwoBodyDecayParameters & param,
									   const std::vector< bool > & selector ) const;

private:

  /**Derivatives of the lab frame momenta of the secondaries w.r.t. px of the primary particle.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdpx( const TwoBodyDecayParameters & param ) const;

  /**Derivatives of the lab frame momenta of the secondaries w.r.t. py of the primary particle.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdpy( const TwoBodyDecayParameters & param ) const;

  /**Derivatives of the lab frame momenta of the secondaries w.r.t. pz of the primary particle.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdpz( const TwoBodyDecayParameters & param ) const;

  /**Derivatives of the lab frame momenta of the secondaries w.r.t. the decay angle theta in the primary's rest frame.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdtheta( const TwoBodyDecayParameters & param ) const;

  /**Derivatives of the lab frame momenta of the secondaries w.r.t. the decay angle phi in the primary's rest frame.
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdphi( const TwoBodyDecayParameters & param ) const;

  /**Derivatives of the lab frame momenta of the secondaries w.r.t. the mass of the primary
   */
  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdm( const TwoBodyDecayParameters & param ) const;

  const std::pair< AlgebraicMatrix, AlgebraicMatrix > dqsdzi( const TwoBodyDecayParameters & param, const DerivativeParameterName & i ) const;

  double thePrimaryMass;
  double theSecondaryMass;

};

#endif
