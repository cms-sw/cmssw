#include "Alignment/KalmanAlignmentAlgorithm/interface/CurrentAlignmentKFUpdator.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "Geometry/Surface/interface/BoundPlane.h"



TrajectoryStateOnSurface CurrentAlignmentKFUpdator::update( const TrajectoryStateOnSurface & tsos,
							    const TransientTrackingRecHit & aRecHit ) const
{
  //std::cout << "[CurrentAlignmentKFUpdator::update] Start Updating." << std::endl;

  double pzSign = tsos.localParameters().pzSign();

  MeasurementExtractor me( tsos );

  AlgebraicVector vecX( tsos.localParameters().vector() );
  AlgebraicSymMatrix matC( tsos.localError().matrix() );
  // Measurement matrix
  AlgebraicMatrix matH( aRecHit.projectionMatrix() );

  // Residuals of aPredictedState w.r.t. aRecHit, 
  AlgebraicVector vecR( aRecHit.parameters() - me.measuredParameters( aRecHit ) );

  // and covariance matrix of residuals
  AlgebraicSymMatrix matV( aRecHit.parametersError() );

  // add information from current estimate on the misalignment
  includeCurrentAlignmentEstimate( aRecHit, tsos, vecR, matV );

  AlgebraicSymMatrix matR( matV + me.measuredError( aRecHit ) );

  int checkInversion = 0;
  AlgebraicSymMatrix invR = matR.inverse( checkInversion );
  if ( checkInversion != 0 )
  {
    std::cout << "[CurrentAlignmentKFUpdator::update] Inversion of matrix R failed." << std::endl;
    return TrajectoryStateOnSurface();
  }

  // Compute Kalman gain matrix
  AlgebraicMatrix matK( matC*matH.T()*invR );

  // Compute local filtered state vector
  AlgebraicVector fsv( vecX + matK*vecR );

  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix matI( 5, 1 );
  AlgebraicMatrix matM( matI - matK*matH );
  AlgebraicSymMatrix fse( matC.similarity( matM ) + matV.similarity( matK ) );

  return TrajectoryStateOnSurface( LocalTrajectoryParameters( fsv, pzSign ), LocalTrajectoryError( fse ),
				   tsos.surface(),&( tsos.globalParameters().magneticField() ) );
}


void CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate( const TransientTrackingRecHit & aRecHit,
								 const TrajectoryStateOnSurface & tsos,
								 AlgebraicVector & vecR,
								 AlgebraicSymMatrix & matV ) const
{
  AlignableDet* alignableDet = theAlignableNavigator->alignableDetFromGeomDet( aRecHit.det() );
  if ( !alignableDet )
  {
    std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] No AlignableDet associated with RecHit." << std::endl;
    return;
  }

  AlignmentParameters* alignmentParameters = getAlignmentParameters( alignableDet );

  if ( alignmentParameters )
  {
    AlgebraicMatrix selectedDerivatives = alignmentParameters->selectedDerivatives( tsos, alignableDet );
    AlgebraicVector selectedParameters = alignmentParameters->selectedParameters();
    AlgebraicSymMatrix selectedCovariance = alignmentParameters->selectedCovariance();

    AlgebraicSymMatrix deltaV = selectedCovariance.similarityT( selectedDerivatives );
    AlgebraicVector deltaR = selectedDerivatives.T()*selectedParameters;

    //AlignmentUserVariables* auv = alignmentParameters->userVariables();
    //if ( !auv ) std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] No AlignmentUserVariables associated with AlignableDet." << std::endl;
    //if ( theAnnealing ) matV *= (*theAnnealing)( auv );

    if ( deltaR.num_row() == vecR.num_row() )
    {
      vecR += deltaR;
      matV += deltaV;
    }
    else std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] Predicted state and misalignment correction not compatible." << std::endl;
  } else std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] No AlignmentParameters associated with AlignableDet." << std::endl;

  return;
}


AlignmentParameters* CurrentAlignmentKFUpdator::getAlignmentParameters( const AlignableDet* alignableDet ) const
{
  // Get alignment parameters from AlignableDet ...
  AlignmentParameters* alignmentParameters = alignableDet->alignmentParameters();
  // ... or any higher level alignable.
  if ( !alignmentParameters ) alignmentParameters = getHigherLevelParameters( alignableDet );
  return alignmentParameters;
}


AlignmentParameters* CurrentAlignmentKFUpdator::getHigherLevelParameters( const Alignable* aAlignable ) const
{
  Alignable* higherLevelAlignable = aAlignable->mother();
  // Alignable has no mother ... most probably the alignable is already the full tracker.
  if ( !higherLevelAlignable ) return 0;
  AlignmentParameters* higherLevelParameters = higherLevelAlignable->alignmentParameters();
  // Found alignment parameters? If not, go one level higher in the hierarchy.
  return higherLevelParameters ? higherLevelParameters : getHigherLevelParameters( higherLevelAlignable );
}
