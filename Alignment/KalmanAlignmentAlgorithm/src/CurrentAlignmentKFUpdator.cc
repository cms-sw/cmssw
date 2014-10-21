#include "Alignment/KalmanAlignmentAlgorithm/interface/CurrentAlignmentKFUpdator.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"


TrajectoryStateOnSurface CurrentAlignmentKFUpdator::update( const TrajectoryStateOnSurface & tsos,
							    const TrackingRecHit & aRecHit ) const 
{
    switch (aRecHit.dimension()) {
        case 1: return update<1>(tsos,aRecHit);
        case 2: return update<2>(tsos,aRecHit);
        case 3: return update<3>(tsos,aRecHit);
        case 4: return update<4>(tsos,aRecHit);
        case 5: return update<5>(tsos,aRecHit);
    }
    throw cms::Exception("Rec hit of invalid dimension (not 1,2,3,4,5)");
}


template <unsigned int D>
TrajectoryStateOnSurface CurrentAlignmentKFUpdator::update( const TrajectoryStateOnSurface & tsos,
							    const TrackingRecHit & aRecHit ) const
{
  //std::cout << "[CurrentAlignmentKFUpdator::update] Start Updating." << std::endl;
  typedef typename AlgebraicROOTObject<D,5>::Matrix MatD5;
  typedef typename AlgebraicROOTObject<5,D>::Matrix Mat5D;
  typedef typename AlgebraicROOTObject<D,D>::SymMatrix SMatDD;
  typedef typename AlgebraicROOTObject<D>::Vector VecD;

  double pzSign = tsos.localParameters().pzSign();

  MeasurementExtractor me( tsos );

  AlgebraicVector5 vecX( tsos.localParameters().vector() );
  AlgebraicSymMatrix55 matC( tsos.localError().matrix() );
  // Measurement matrix
  MatD5 matH = asSMatrix<D,5>( aRecHit.projectionMatrix() );

  // Residuals of aPredictedState w.r.t. aRecHit, 
  VecD vecR = asSVector<D>(aRecHit.parameters()) - me.measuredParameters<D>( aRecHit );

  // and covariance matrix of residuals
  SMatDD matV = asSMatrix<D>( aRecHit.parametersError() );

  // add information from current estimate on the misalignment
  includeCurrentAlignmentEstimate<D>( aRecHit, tsos, vecR, matV );

   SMatDD  matR( matV + me.measuredError<D>( aRecHit ) );

  int checkInversion = 0;
  SMatDD invR = matR.Inverse( checkInversion );
  if ( checkInversion != 0 )
  {
    std::cout << "[CurrentAlignmentKFUpdator::update] Inversion of matrix R failed." << std::endl;
    return TrajectoryStateOnSurface();
  }

  // Compute Kalman gain matrix
  Mat5D matK =  matC*ROOT::Math::Transpose(matH)*invR ;

  // Compute local filtered state vector
  AlgebraicVector5 fsv( vecX + matK*vecR );

  // Compute covariance matrix of local filtered state vector
  AlgebraicSymMatrix55 matI  = AlgebraicMatrixID();
  AlgebraicMatrix55 matM( matI - matK*matH );
  AlgebraicSymMatrix55 fse( ROOT::Math::Similarity(matM, matC) + ROOT::Math::Similarity(matK, matV) );

  return TrajectoryStateOnSurface( LocalTrajectoryParameters( fsv, pzSign ), LocalTrajectoryError( fse ),
				   tsos.surface(),&( tsos.globalParameters().magneticField() ) );
}


template <unsigned int D>
void CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate( const TrackingRecHit & aRecHit,
								 const TrajectoryStateOnSurface & tsos,
								 typename AlgebraicROOTObject<D>::Vector & vecR,
								 typename AlgebraicROOTObject<D>::SymMatrix & matV ) const
{
  const GeomDet* det = aRecHit.det();
  if ( !det ) return;

  AlignableDetOrUnitPtr alignableDet = theAlignableNavigator->alignableFromGeomDet( det );
  if ( alignableDet.isNull() )
  {
    //std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] No AlignableDet associated with RecHit." << std::endl;
    return;
  }

  AlignmentParameters const* alignmentParameters = getAlignmentParameters( alignableDet );

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

    if ( deltaR.num_row() == D )
    {
      vecR += asSVector<D>(deltaR);
      matV += asSMatrix<D>(deltaV);
    }
    else std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] Predicted state and misalignment correction not compatible." << std::endl;
  } else std::cout << "[CurrentAlignmentKFUpdator::includeCurrentAlignmentEstimate] No AlignmentParameters associated with AlignableDet." << std::endl;

  return;
}


AlignmentParameters const* CurrentAlignmentKFUpdator::getAlignmentParameters( const AlignableDetOrUnitPtr& alignableDet ) const
{
  // Get alignment parameters from AlignableDet ...
  AlignmentParameters const* alignmentParameters = alignableDet->alignmentParameters();
  // ... or any higher level alignable.
  if ( !alignmentParameters ) alignmentParameters = getHigherLevelParameters( alignableDet );
  return alignmentParameters;
}


AlignmentParameters const* CurrentAlignmentKFUpdator::getHigherLevelParameters( const Alignable* aAlignable ) const
{
  Alignable* higherLevelAlignable = aAlignable->mother();
  // Alignable has no mother ... most probably the alignable is already the full tracker.
  if ( !higherLevelAlignable ) return 0;
  AlignmentParameters* higherLevelParameters = higherLevelAlignable->alignmentParameters();
  // Found alignment parameters? If not, go one level higher in the hierarchy.
  return higherLevelParameters ? higherLevelParameters : getHigherLevelParameters( higherLevelAlignable );
}
