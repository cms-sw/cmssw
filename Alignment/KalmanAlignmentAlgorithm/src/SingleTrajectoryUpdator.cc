
#include "Alignment/KalmanAlignmentAlgorithm/interface/SingleTrajectoryUpdator.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterStore.h" 

#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentDerivativesExtractor.h"

#include "Utilities/Timing/interface/TimingReport.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

using namespace std;


SingleTrajectoryUpdator::SingleTrajectoryUpdator( const edm::ParameterSet & config ) :
  KalmanAlignmentUpdator( config ) {}


SingleTrajectoryUpdator::~SingleTrajectoryUpdator( void ) {}


void SingleTrajectoryUpdator::process( const ReferenceTrajectoryPtr & trajectory,
				       AlignmentParameterStore* store,
				       AlignableNavigator* navigator,
				       KalmanAlignmentMetricsUpdator* metrics )
{
  if ( !( *trajectory ).isValid() ) return;

  TimeMe* timer;
  timer = new TimeMe( "Retrieve_Alignables" );

  vector< AlignableDetOrUnitPtr > currentAlignableDets = navigator->alignablesFromHits( ( *trajectory ).recHits() );
  vector< Alignable* > currentAlignables = alignablesFromAlignableDets( currentAlignableDets, store );

  delete timer;
  timer = new TimeMe( "Update_Metrics" );

  metrics->update( currentAlignables );

  delete timer;
  timer = new TimeMe( "Retrieve_Alignables" );

  vector< Alignable* > additionalAlignables = metrics->additionalAlignables( currentAlignables );

  vector< Alignable* > allAlignables;
  allAlignables.reserve( currentAlignables.size() + additionalAlignables.size() );
  allAlignables.insert( allAlignables.end(), currentAlignables.begin(), currentAlignables.end() );
  allAlignables.insert( allAlignables.end(), additionalAlignables.begin(), additionalAlignables.end() );

  delete timer;
  timer = new TimeMe( "Retrieve_Parameters" );

  CompositeAlignmentParameters alignmentParameters = store->selectParameters( allAlignables );

  delete timer;
  timer = new TimeMe( "Retrieve_Matrices" );

  const AlgebraicVector& allAlignmentParameters = alignmentParameters.parameters();
  AlgebraicSymMatrix currentAlignmentCov = alignmentParameters.covarianceSubset( currentAlignables );
  AlgebraicSymMatrix additionalAlignmentCov = alignmentParameters.covarianceSubset( additionalAlignables );
  AlgebraicMatrix mixedAlignmentCov = alignmentParameters.covarianceSubset( additionalAlignables, currentAlignables );
  AlgebraicMatrix alignmentCovSubset = alignmentParameters.covarianceSubset( alignmentParameters.components(), currentAlignables );

  CompositeAlignmentDerivativesExtractor extractor( currentAlignables, currentAlignableDets, ( *trajectory ).trajectoryStates() );
  AlgebraicVector correctionTerm = extractor.correctionTerm();
  AlgebraicMatrix alignmentDeriv = extractor.derivatives();
  AlgebraicSymMatrix alignmentCov = currentAlignmentCov.similarity( alignmentDeriv );

  AlgebraicVector allMeasurements = ( *trajectory ).measurements();
  AlgebraicSymMatrix measurementCov = ( *trajectory ).measurementErrors();
  AlgebraicVector referenceTrajectory = ( *trajectory ).trajectoryPositions();
  AlgebraicMatrix derivatives = ( *trajectory ).derivatives();

  //measurementCov += 1e-4*AlgebraicSymMatrix( measurementCov.num_row(), 1 );

  delete timer;
  timer = new TimeMe( "Update_Algo" );

  AlgebraicSymMatrix misalignedCov = measurementCov + alignmentCov;

  int checkInversion = 0;

  AlgebraicSymMatrix invMisalignedCov = misalignedCov.inverse( checkInversion );
  if ( checkInversion != 0 )
  {
    cout << "[KalmanAlignment] WARNING: 'AlgebraicSymMatrix misalignedCov' not invertible." << endl;
    return;
  }
  AlgebraicSymMatrix limitCov1 = ( invMisalignedCov.similarityT( derivatives ) ).inverse( checkInversion );
  if ( checkInversion != 0 )
  {
    cout << "[KalmanAlignment] WARNING: 'AlgebraicSymMatrix limitCov1' not computed." << endl;
    return;
  }
  AlgebraicSymMatrix limitCov2 = limitCov1.similarity( invMisalignedCov*derivatives );
  AlgebraicSymMatrix limitCov = invMisalignedCov - limitCov2;
  AlgebraicMatrix fullCovTimesDeriv = alignmentCovSubset*alignmentDeriv.T();
  AlgebraicMatrix fullGainMatrix = fullCovTimesDeriv*limitCov;
  AlgebraicMatrix covTimesDeriv = currentAlignmentCov*alignmentDeriv.T();
  AlgebraicMatrix gainMatrix = covTimesDeriv*limitCov;

  // make updates for the kalman-filter
  // update of parameters
  AlgebraicVector updatedAlignmentParameters =
    allAlignmentParameters + fullGainMatrix*( allMeasurements - correctionTerm - referenceTrajectory );

  // update of covariance
  int nCRow = currentAlignmentCov.num_row();
  int nARow = additionalAlignmentCov.num_row();

  AlgebraicSymMatrix updatedAlignmentCov( nCRow + nARow );

  AlgebraicMatrix gTimesDeriv = limitCov*alignmentDeriv;
  AlgebraicMatrix simMat = AlgebraicMatrix( nCRow, nCRow, 1 ) - covTimesDeriv*gTimesDeriv;
  AlgebraicSymMatrix updatedCurrentAlignmentCov = currentAlignmentCov.similarity( simMat ) + measurementCov.similarity( gainMatrix );

  AlgebraicMatrix mixedUpdateMat = simMat.T()*simMat.T() + measurementCov.similarity( gTimesDeriv.T() )*currentAlignmentCov;
  AlgebraicMatrix updatedMixedAlignmentCov = mixedAlignmentCov*mixedUpdateMat;

  AlgebraicSymMatrix additionalUpdateMat = misalignedCov.similarity( gTimesDeriv.T() ) - 2.*limitCov.similarity( alignmentDeriv.T() );
  AlgebraicSymMatrix updatedAdditionalAlignmentCov = additionalAlignmentCov + additionalUpdateMat.similarity( mixedAlignmentCov );

  for ( int nRow=0; nRow<nCRow; nRow++ )
  {
    for ( int nCol=0; nCol<=nRow; nCol++ ) updatedAlignmentCov[nRow][nCol] = updatedCurrentAlignmentCov[nRow][nCol];
  }

  for ( int nRow=0; nRow<nARow; nRow++ )
  {
     for ( int nCol=0; nCol<=nRow; nCol++ ) updatedAlignmentCov[nRow+nCRow][nCol+nCRow] = updatedAdditionalAlignmentCov[nRow][nCol];
  }

  for ( int nRow=0; nRow<nARow; nRow++ )
  {
    for ( int nCol=0; nCol<nCRow; nCol++ ) updatedAlignmentCov[nRow+nCRow][nCol] = updatedMixedAlignmentCov[nRow][nCol];
  }

  if ( !checkCovariance( updatedAlignmentCov ) ) throw cms::Exception( "LogicError" );

  delete timer;
  timer = new TimeMe( "Clone_Parameters" );

  // update in alignment-interface
  CompositeAlignmentParameters* updatedParameters;
  updatedParameters = alignmentParameters.clone( updatedAlignmentParameters, updatedAlignmentCov );

  delete timer;
  timer = new TimeMe( "Update_Parameters" );

  store->updateParameters( *updatedParameters );
  delete updatedParameters;

  delete timer;
  timer = new TimeMe( "Update_UserVariables" );

  // update user variables for debugging
  //updateUserVariables( alignmentParameters.components() );
  updateUserVariables( currentAlignables );

  delete timer;

  return;
}


bool SingleTrajectoryUpdator::checkCovariance( const AlgebraicSymMatrix& cov ) const
{
  for ( int i = 0; i < cov.num_row(); ++i )
  {
    if ( cov[i][i] < 0. ) return false;
  }

  return true;
}
