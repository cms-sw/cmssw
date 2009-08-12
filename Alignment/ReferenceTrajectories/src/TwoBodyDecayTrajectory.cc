
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectory.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/Math/interface/Error.h" 

#include "Geometry/CommonDetUnit/interface/GeomDet.h"


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

// Break Points not implemented
TwoBodyDecayTrajectory::TwoBodyDecayTrajectory( const TwoBodyDecayTrajectoryState& trajectoryState,
						const ConstRecHitCollection & recHits,
						const MagneticField* magField,
						MaterialEffects materialEffects,
						PropagationDirection propDir,
						bool hitsAreReverse,
						bool useRefittedState,
						bool constructTsosWithErrors )

  : ReferenceTrajectoryBase( TwoBodyDecayParameters::dimension, recHits.first.size() + recHits.second.size(), 0 )
{
  if ( hitsAreReverse )
  {
    TransientTrackingRecHit::ConstRecHitContainer::const_reverse_iterator itRecHits;
    ConstRecHitCollection fwdRecHits;

    fwdRecHits.first.reserve( recHits.first.size() );
    for ( itRecHits = recHits.first.rbegin(); itRecHits != recHits.first.rend(); ++itRecHits )
    {
      fwdRecHits.first.push_back( *itRecHits );
    }

    fwdRecHits.second.reserve( recHits.second.size() );
    for ( itRecHits = recHits.second.rbegin(); itRecHits != recHits.second.rend(); ++itRecHits )
    {
      fwdRecHits.second.push_back( *itRecHits );
    }

    theValidityFlag = this->construct( trajectoryState, fwdRecHits, magField, materialEffects, propDir,
				       useRefittedState, constructTsosWithErrors );
  }
  else
  {
    theValidityFlag = this->construct( trajectoryState, recHits, magField, materialEffects, propDir,
				       useRefittedState, constructTsosWithErrors );
  }
}


TwoBodyDecayTrajectory::TwoBodyDecayTrajectory( void )
  : ReferenceTrajectoryBase( 0, 0, 0 )
{}


bool TwoBodyDecayTrajectory::construct( const TwoBodyDecayTrajectoryState& state,
					const ConstRecHitCollection& recHits,
					const MagneticField* field,
					MaterialEffects materialEffects,
					PropagationDirection propDir,
					bool useRefittedState,
					bool constructTsosWithErrors )
{
  const TwoBodyDecayTrajectoryState::TsosContainer& tsos = state.trajectoryStates( useRefittedState );
  const TwoBodyDecayTrajectoryState::Derivatives& deriv = state.derivatives();
  double mass = state.particleMass();

  //
  // first track
  //

  // construct a trajectory (hits should be already in correct order)
  ReferenceTrajectory trajectory1( tsos.first, recHits.first, false, field, (materialEffects ==  breakPoints) ? combined : materialEffects, propDir, mass );

  // check if construction of trajectory was successful
  if ( !trajectory1.isValid() ) return false;

  // derivatives of the trajectory w.r.t. to the decay parameters
  AlgebraicMatrix fullDeriv1 = trajectory1.derivatives()*deriv.first;

  //
  // second track
  //

  ReferenceTrajectory trajectory2( tsos.second, recHits.second, false, field, (materialEffects ==  breakPoints) ? combined : materialEffects, propDir, mass );

  if ( !trajectory2.isValid() ) return false;

  AlgebraicMatrix fullDeriv2 = trajectory2.derivatives()*deriv.second;

  //
  // combine both tracks
  //

  theNumberOfRecHits.first = recHits.first.size();
  theNumberOfRecHits.second = recHits.second.size();

  int nMeasurements1 = nMeasPerHit*theNumberOfRecHits.first;
  //int nMeasurements2 = nMeasPerHit*theNumberOfRecHits.second;
  //int nMeasurements = nMeasurements1 + nMeasurements2;

  theDerivatives.sub( 1, 1, fullDeriv1 );
  theDerivatives.sub( nMeasurements1 + 1, 1, fullDeriv2 );

  theMeasurements.sub( 1, trajectory1.measurements() );
  theMeasurements.sub( nMeasurements1 + 1, trajectory2.measurements() );

  theMeasurementsCov.sub( 1, trajectory1.measurementErrors() );
  theMeasurementsCov.sub( nMeasurements1 + 1, trajectory2.measurementErrors() );

  theTrajectoryPositions.sub( 1, trajectory1.trajectoryPositions() );
  theTrajectoryPositions.sub( nMeasurements1 + 1, trajectory2.trajectoryPositions() );

  theTrajectoryPositionCov = state.decayParameters().covariance().similarity( theDerivatives );

  theParameters = state.decayParameters().parameters();

  theRecHits.insert( theRecHits.end(), recHits.first.begin(), recHits.first.end() );
  theRecHits.insert( theRecHits.end(), recHits.second.begin(), recHits.second.end() );

  if ( constructTsosWithErrors )
  {
    constructTsosVecWithErrors( trajectory1, trajectory2, field );
  }
  else
  {
    theTsosVec.insert( theTsosVec.end(),
		       trajectory1.trajectoryStates().begin(),
		       trajectory1.trajectoryStates().end() );

    theTsosVec.insert( theTsosVec.end(),
		       trajectory2.trajectoryStates().begin(),
		       trajectory2.trajectoryStates().end() );
  }

  return true;
}


void TwoBodyDecayTrajectory::constructTsosVecWithErrors( const ReferenceTrajectory& traj1,
							 const ReferenceTrajectory& traj2,
							 const MagneticField* field )
{
  int iTsos = 0;

  std::vector< TrajectoryStateOnSurface >::const_iterator itTsos;

  for ( itTsos = traj1.trajectoryStates().begin(); itTsos != traj1.trajectoryStates().end(); itTsos++ )
  {
    constructSingleTsosWithErrors( *itTsos, iTsos, field );
    iTsos++;
  }

  for ( itTsos = traj2.trajectoryStates().begin(); itTsos != traj2.trajectoryStates().end(); itTsos++ )
  {
    constructSingleTsosWithErrors( *itTsos, iTsos, field );
    iTsos++;
  }
}


void TwoBodyDecayTrajectory::constructSingleTsosWithErrors( const TrajectoryStateOnSurface& tsos,
							    int iTsos,
							    const MagneticField* field )
{
  AlgebraicSymMatrix localErrors( 5, 0 );
  const double coeff = 1e-2;

  double invP = tsos.localParameters().signedInverseMomentum();
  LocalVector p = tsos.localParameters().momentum();

  // rough estimate for the errors of q/p, dx/dz and dy/dz, assuming that
  // sigma(px) = sigma(py) = sigma(pz) = coeff*p.
  float dpinv = coeff*( fabs(p.x()) + fabs(p.y()) + fabs(p.z()) )*invP*invP;
  float dxdir = coeff*( fabs(p.x()) + fabs(p.z()) )/p.z()/p.z();
  float dydir = coeff*( fabs(p.y()) + fabs(p.z()) )/p.z()/p.z();
  localErrors[0][0] = dpinv*dpinv;
  localErrors[1][1] = dxdir*dxdir;
  localErrors[2][2] = dydir*dydir;

  // exact values for the errors on local x and y
  localErrors[3][3] = theTrajectoryPositionCov[nMeasPerHit*iTsos][nMeasPerHit*iTsos];
  localErrors[3][4] = theTrajectoryPositionCov[nMeasPerHit*iTsos][nMeasPerHit*iTsos+1];
  localErrors[4][4] = theTrajectoryPositionCov[nMeasPerHit*iTsos+1][nMeasPerHit*iTsos+1];

  // construct tsos with local errors
  theTsosVec[iTsos] =  TrajectoryStateOnSurface( tsos.localParameters(),
						 LocalTrajectoryError( localErrors ),
						 tsos.surface(),
						 field,
						 tsos.surfaceSide() );
}
