
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayDerivatives.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "DataFormats/CLHEP/interface/Migration.h"

using namespace std;


TwoBodyDecayTrajectoryState::TwoBodyDecayTrajectoryState( const TsosContainer & tsos,
							  const TwoBodyDecay & tbd,
							  double particleMass,
							  const MagneticField* magField,
							  bool propagateErrors )
  : theValidityFlag( false ),
    theParticleMass( particleMass ),
    theParameters( tbd.decayParameters() ),
    theDerivatives( AlgebraicMatrix( nLocalParam, nDecayParam ), AlgebraicMatrix( nLocalParam, nDecayParam ) ),
    theOriginalTsos( tsos ),
    thePrimaryMass( tbd.primaryMass() ),
    thePrimaryWidth( tbd.primaryWidth() )
{
  construct( magField, propagateErrors );
}


TwoBodyDecayTrajectoryState::TwoBodyDecayTrajectoryState( const TsosContainer & tsos,
							  const TwoBodyDecayParameters & param,
							  double particleMass,
							  const MagneticField* magField,
							  bool propagateErrors )
  : theValidityFlag( false ),
    theParticleMass( particleMass ),
    theParameters( param ),
    theDerivatives( AlgebraicMatrix( 5, 9 ), AlgebraicMatrix( 5, 9 ) ),
    theOriginalTsos( tsos ),
    thePrimaryMass( 0. ),
    thePrimaryWidth( -1. ) // dummy values
{
  construct( magField, propagateErrors );
}


void TwoBodyDecayTrajectoryState::rescaleError( double scale )
{
  theOriginalTsos.first.rescaleError( scale );
  theOriginalTsos.second.rescaleError( scale );
  theRefittedTsos.first.rescaleError( scale );
  theRefittedTsos.second.rescaleError( scale );
}


void TwoBodyDecayTrajectoryState::construct( const MagneticField* magField,
					     bool propagateErrors )
{
  // construct global trajectory parameters at the starting point
  TwoBodyDecayModel tbdDecayModel( theParameters[TwoBodyDecayParameters::mass], theParticleMass );
  pair< AlgebraicVector, AlgebraicVector > secondaryMomenta = tbdDecayModel.cartesianSecondaryMomenta( theParameters );

  GlobalPoint vtx( theParameters[TwoBodyDecayParameters::x],
		   theParameters[TwoBodyDecayParameters::y],
		   theParameters[TwoBodyDecayParameters::z] );

  GlobalVector p1( secondaryMomenta.first[0],
		   secondaryMomenta.first[1],
		   secondaryMomenta.first[2] );

  GlobalVector p2( secondaryMomenta.second[0],
		   secondaryMomenta.second[1],
		   secondaryMomenta.second[2] );

  GlobalTrajectoryParameters gtp1( vtx, p1, theOriginalTsos.first.charge(), magField );
  FreeTrajectoryState fts1( gtp1 );

  GlobalTrajectoryParameters gtp2( vtx, p2, theOriginalTsos.second.charge(), magField );
  FreeTrajectoryState fts2( gtp2 );

  // contruct derivatives at the starting point
  TwoBodyDecayDerivatives tbdDerivatives( theParameters[TwoBodyDecayParameters::mass], theParticleMass );
  pair< AlgebraicMatrix, AlgebraicMatrix > derivatives = tbdDerivatives.derivatives( theParameters );

  AlgebraicMatrix deriv1( 6, 9, 0 );
  deriv1.sub( 1, 1, AlgebraicMatrix( 3, 3, 1 ) );
  deriv1.sub( 4, 4, derivatives.first );

  AlgebraicMatrix deriv2( 6, 9, 0 );
  deriv2.sub( 1, 1, AlgebraicMatrix( 3, 3, 1 ) );
  deriv2.sub( 4, 4, derivatives.second );

  // compute errors of initial states
  if ( propagateErrors ) {
    setError( fts1, deriv1 );
    setError( fts2, deriv2 );
  }


  // propgate states and derivatives from the starting points to the end points
  bool valid1 = propagateSingleState( fts1, gtp1, deriv1, theOriginalTsos.first.surface(),
				      magField, theRefittedTsos.first, theDerivatives.first );

  bool valid2 = propagateSingleState( fts2, gtp2, deriv2, theOriginalTsos.second.surface(),
				      magField, theRefittedTsos.second, theDerivatives.second );

  theValidityFlag = valid1 && valid2;

  return;
}


bool TwoBodyDecayTrajectoryState::propagateSingleState( const FreeTrajectoryState & fts,
							const GlobalTrajectoryParameters & gtp,
							const AlgebraicMatrix & startDeriv,
							const Surface & surface,
							const MagneticField* magField,
							TrajectoryStateOnSurface & tsos,
							AlgebraicMatrix & endDeriv ) const
{
  AnalyticalPropagator propagator( magField );

  // propagate state
  pair< TrajectoryStateOnSurface, double > tsosWithPath = propagator.propagateWithPath( fts, surface );

  // check if propagation was successful
  if ( !tsosWithPath.first.isValid() ) return false;

  // jacobian for transformation from cartesian to curvilinear frame at the starting point
  JacobianCartesianToCurvilinear cartToCurv( gtp );
  const AlgebraicMatrix56& matCartToCurv = cartToCurv.jacobian();

  // jacobian in curvilinear frame for propagation from the starting point to the end point
  AnalyticalCurvilinearJacobian curvJac( gtp, tsosWithPath.first.globalPosition(),
					 tsosWithPath.first.globalMomentum(),
					 tsosWithPath.second );
  const AlgebraicMatrix55& matCurvJac = curvJac.jacobian();

  // jacobian for transformation from curvilinear to local frame at the end point
  JacobianCurvilinearToLocal curvToLoc( surface, tsosWithPath.first.localParameters(), *magField );
  const AlgebraicMatrix55& matCurvToLoc = curvToLoc.jacobian();

  AlgebraicMatrix56 tmpDeriv = matCurvToLoc*matCurvJac*matCartToCurv;
  AlgebraicMatrix hepMatDeriv( asHepMatrix( tmpDeriv ) );
  //AlgebraicMatrix hepMatDeriv = asHepMatrix< 5, 6 >( tmpDeriv );

  // replace original state with new state
  tsos = tsosWithPath.first;

  // propagate derivative matrix
  endDeriv = hepMatDeriv*startDeriv;

  return true;
}


void TwoBodyDecayTrajectoryState::setError( FreeTrajectoryState& fts,
					    AlgebraicMatrix& derivative ) const
{
  AlgebraicSymMatrix ftsCartesianError( theParameters.covariance().similarity( derivative ) );
  fts.setCartesianError( asSMatrix<6>( ftsCartesianError ) );
}
