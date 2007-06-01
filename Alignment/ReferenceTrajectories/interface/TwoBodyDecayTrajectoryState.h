#ifndef Alignment_ReferenceTrajectories_TwoBodyDecayTrajectoryState_h
#define Alignment_ReferenceTrajectories_TwoBodyDecayTrajectoryState_h

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecay.h"

/** Computes the trajectory states and their derivatives w.r.t. the decay parameters.
 */

class GlobalTrajectoryParameters;


class TwoBodyDecayTrajectoryState
{

public:

  typedef std::pair< TrajectoryStateOnSurface, TrajectoryStateOnSurface > TsosContainer;
  typedef std::pair< AlgebraicMatrix, AlgebraicMatrix > Derivatives;

  /** The constructor takes the two trajectory states that are to be updated (typically the
   *  innermost trajectory states of two tracks) and the decay parameters.
   */
  TwoBodyDecayTrajectoryState( const TsosContainer & tsos,
			       const TwoBodyDecay & tbd,
			       double particleMass,
			       const MagneticField* magField );

  /** The constructor takes the two trajectory states that are to be updated (typically the
   *  innermost trajectory states of two tracks) and the decay parameters.
   */
  TwoBodyDecayTrajectoryState( const TsosContainer & tsos,
			       const TwoBodyDecayParameters & param,
			       double particleMass,
			       const MagneticField* magField );

  ~TwoBodyDecayTrajectoryState( void ) {}

  inline bool isValid( void ) const { return theValidityFlag; }

  inline double particleMass( void ) const { return theParticleMass; }
  inline const TwoBodyDecayParameters & decayParameters( void ) const { return theParameters; }
  inline const TsosContainer& trajectoryStates( bool useRefittedState = true ) const { return useRefittedState ? theRefittedTsos : theOriginalTsos; }
  inline const Derivatives& derivatives( void ) const { return theDerivatives; }

private:

  void construct( const MagneticField* magField );

  bool propagateSingleState( const GlobalTrajectoryParameters & gtp,
			     const AlgebraicMatrix & startDeriv,
			     const Surface & surface,
			     const MagneticField* magField,
			     TrajectoryStateOnSurface & tsos,
			     AlgebraicMatrix & endDeriv );

  bool theValidityFlag;

  double theParticleMass;

  TwoBodyDecayParameters theParameters;
  Derivatives theDerivatives;
  TsosContainer theOriginalTsos;
  TsosContainer theRefittedTsos;


  static const unsigned int nLocalParam = 5;
  static const unsigned int nDecayParam = TwoBodyDecayParameters::dimension;
};


#endif
