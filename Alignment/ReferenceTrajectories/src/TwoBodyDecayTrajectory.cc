
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectory.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/Math/interface/Error.h" 

#include "Geometry/CommonDetUnit/interface/GeomDet.h"


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

TwoBodyDecayTrajectory::TwoBodyDecayTrajectory( const TwoBodyDecayTrajectoryState& trajectoryState,
						const ConstRecHitCollection & recHits,
						const MagneticField* magField,
						MaterialEffects materialEffects,
						PropagationDirection propDir,
						bool hitsAreReverse,
						const reco::BeamSpot &beamSpot,
						bool useRefittedState,
						bool constructTsosWithErrors )

  : ReferenceTrajectoryBase( 
     TwoBodyDecayParameters::dimension, recHits.first.size() + recHits.second.size(),
  (materialEffects >= breakPoints) ? 2*(recHits.first.size() + recHits.second.size())-4 : 0,
  (materialEffects >= breakPoints) ? 2*(recHits.first.size() + recHits.second.size())-4 : 0 )
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
				       beamSpot, useRefittedState, constructTsosWithErrors );
  }
  else
  {
    theValidityFlag = this->construct( trajectoryState, recHits, magField, materialEffects, propDir,
				       beamSpot, useRefittedState, constructTsosWithErrors );
  }
}


TwoBodyDecayTrajectory::TwoBodyDecayTrajectory( void )
  : ReferenceTrajectoryBase( 0, 0, 0, 0)
{}


bool TwoBodyDecayTrajectory::construct( const TwoBodyDecayTrajectoryState& state,
					const ConstRecHitCollection& recHits,
					const MagneticField* field,
					MaterialEffects materialEffects,
					PropagationDirection propDir,
					const reco::BeamSpot &beamSpot,
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
  ReferenceTrajectory trajectory1( tsos.first, recHits.first, false, field, materialEffects,
				   propDir, mass, false, beamSpot);

  // check if construction of trajectory was successful
  if ( !trajectory1.isValid() ) return false;
  
  int nLocal = deriv.first.num_row();
  int nTbd   = deriv.first.num_col();
  unsigned int nHitMeas1 = trajectory1.numberOfHitMeas();
  unsigned int nMsMeas1  = trajectory1.numberOfMsMeas(); 
  unsigned int nPar1     = trajectory1.numberOfPar();
  unsigned int nMsPar1   = trajectory1.numberOfMsPar(); 
     
  // derivatives of the trajectory w.r.t. to the decay parameters
  AlgebraicMatrix fullDeriv1 = trajectory1.derivatives().sub(1,nHitMeas1+nMsMeas1,1,nLocal) * trajectory1.localToTrajectory() * deriv.first;

  //
  // second track
  //

  ReferenceTrajectory trajectory2( tsos.second, recHits.second, false, field, materialEffects,
				   propDir, mass, false, beamSpot );

  if ( !trajectory2.isValid() ) return false;
  
  unsigned int nHitMeas2 = trajectory2.numberOfHitMeas();
  unsigned int nMsMeas2  = trajectory2.numberOfMsMeas();  
  unsigned int nPar2     = trajectory2.numberOfPar();
  unsigned int nMsPar2   = trajectory2.numberOfMsPar();

  AlgebraicMatrix fullDeriv2 = trajectory2.derivatives().sub(1,nHitMeas2+nMsMeas2,1,nLocal) * trajectory2.localToTrajectory() * deriv.second;

  //
  // combine both tracks
  //
  
  theNumberOfRecHits.first = recHits.first.size();
  theNumberOfRecHits.second = recHits.second.size();

  theNumberOfHits = trajectory1.numberOfHits() + trajectory2.numberOfHits(); 
  theNumberOfPars = nPar1 + nPar2;
  theNumberOfMsPars = nMsPar1 + nMsPar2;
  theNumberOfMsMeas = nMsMeas1 + nMsMeas2;
  
  // hit measurements from trajectory 1
  int rowOffset = 1;
  int colOffset = 1; 
  theDerivatives.sub( rowOffset, colOffset,                fullDeriv1.sub(            1, nHitMeas1,                     1, nTbd ) );
  colOffset += nTbd;
  theDerivatives.sub( rowOffset, colOffset, trajectory1.derivatives().sub(            1, nHitMeas1,            nLocal + 1, nPar1 + nMsPar1 ) );
  // hit measurements from trajectory 2
  rowOffset += nHitMeas1;
  colOffset = 1; 
  theDerivatives.sub( rowOffset, colOffset,                fullDeriv2.sub(            1, nHitMeas2,                     1, nTbd ) );
  colOffset += (nPar1 + nMsPar1 + nTbd - nLocal);
  theDerivatives.sub( rowOffset, colOffset, trajectory2.derivatives().sub(            1, nHitMeas2,            nLocal + 1, nPar2 + nMsPar2 ) );  
  // MS measurements from trajectory 1
  rowOffset += nHitMeas2;  
  colOffset = 1; 
  theDerivatives.sub( rowOffset, colOffset,                fullDeriv1.sub(nHitMeas1 + 1, nHitMeas1 + nMsMeas1,          1, nTbd ) );  
  colOffset += nTbd;
  theDerivatives.sub( rowOffset, colOffset, trajectory1.derivatives().sub(nHitMeas1 + 1, nHitMeas1 + nMsMeas1, nLocal + 1, nPar1 + nMsPar1 ) ); 
  // MS measurements from trajectory 2
  rowOffset += nMsMeas1;  
  colOffset = 1; 
  theDerivatives.sub( rowOffset, colOffset,                fullDeriv2.sub(nHitMeas2 + 1, nHitMeas2 + nMsMeas2,          1, nTbd ) );
  colOffset += (nPar1 + nMsPar1 + nTbd - nLocal);  
  theDerivatives.sub( rowOffset, colOffset, trajectory2.derivatives().sub(nHitMeas2 + 1, nHitMeas2 + nMsMeas2, nLocal + 1, nPar2 + nMsPar2 ) ); 
        
  theMeasurements.sub(                                    1, trajectory1.measurements().sub(            1, nHitMeas1 ) );
  theMeasurements.sub( nHitMeas1                        + 1, trajectory2.measurements().sub(            1, nHitMeas2 ) );
  theMeasurements.sub( nHitMeas1 + nHitMeas2            + 1, trajectory1.measurements().sub(nHitMeas1 + 1, nHitMeas1 + nMsMeas1 ) );
  theMeasurements.sub( nHitMeas1 + nHitMeas2 + nMsMeas1 + 1, trajectory2.measurements().sub(nHitMeas2 + 1, nHitMeas2 + nMsMeas2 ) );

  theMeasurementsCov.sub(                                    1, trajectory1.measurementErrors().sub(            1, nHitMeas1 ) );
  theMeasurementsCov.sub( nHitMeas1                        + 1, trajectory2.measurementErrors().sub(            1, nHitMeas2 ) );
  theMeasurementsCov.sub( nHitMeas1 + nHitMeas2            + 1, trajectory1.measurementErrors().sub(nHitMeas1 + 1, nHitMeas1 + nMsMeas1 ) );
  theMeasurementsCov.sub( nHitMeas1 + nHitMeas2 + nMsMeas1 + 1, trajectory2.measurementErrors().sub(nHitMeas2 + 1, nHitMeas2 + nMsMeas2 ) );

  theTrajectoryPositions.sub(             1, trajectory1.trajectoryPositions() );
  theTrajectoryPositions.sub( nHitMeas1 + 1, trajectory2.trajectoryPositions() );

  theTrajectoryPositionCov = state.decayParameters().covariance().similarity( theDerivatives.sub(1, nHitMeas1 + nHitMeas2, 1, 9) );
  
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
