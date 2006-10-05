
#include "Alignment/KalmanAlignmentAlgorithm/interface/SingleTrajectoryUpdator.h"

using namespace std;


SingleTrajectoryUpdator::SingleTrajectoryUpdator( const edm::ParameterSet & config ) :
  KalmanAlignmentUpdator( config )
{
  theMaxDistance = config.getParameter< int >( "MaxDistance" );
  theMetricsCalculator.setMaxDistance( theMaxDistance );
}


SingleTrajectoryUpdator::~SingleTrajectoryUpdator( void ) {}


void SingleTrajectoryUpdator::process( const ReferenceTrajectoryPtr & trajectory,
				       AlignmentParameterStore* store,
				       AlignableNavigator* navigator )
{
  cout << "[SingleTrajectoryUpdator::process] number of alignables: " << store->alignables().size() << endl;
  cout << "[SingleTrajectoryUpdator::process] trajectory->isValid() = " << trajectory->isValid() << endl;

  if ( !( *trajectory ).isValid() ) return;

  //vector< AlignableDet* > currentAlignableDets = store->alignableDetsFromHits( ( *trajectory ).recHits() );
  //vector< AlignableDet* > currentAlignableDets = navigator->alignableDetsFromHits( ( *trajectory ).recHits() );
  vector< AlignableDet* > currentAlignableDets = alignableDetsFromHits( ( *trajectory ).recHits(), navigator );
  // init with current AlignableDets and add additional AlignableDets later
  vector< AlignableDet* > allAlignableDets = currentAlignableDets;
  vector< AlignableDet* > additionalAlignableDets;
  vector< AlignableDet* >::iterator itAD; 

  // compute distances
  theMetricsCalculator.computeDistances( currentAlignableDets );

  map< AlignableDet*, int > updateList;
  map< AlignableDet*, int >::iterator itUL;

  set< AlignableDet* > alignableDetsFromUpdateList;
  set< AlignableDet* >::iterator itAUL;

  // make union of all lists
  for ( itAD = currentAlignableDets.begin(); itAD != currentAlignableDets.end(); itAD++ )
  {
    updateList = theMetricsCalculator.getDistances( *itAD );
    for ( itUL = updateList.begin(); itUL != updateList.end(); itUL++ )
    {
      if ( itUL->second <= theMaxDistance ) alignableDetsFromUpdateList.insert( itUL->first );
    }
  }

  // make final list of modules for update
  for ( itAUL = alignableDetsFromUpdateList.begin(); itAUL != alignableDetsFromUpdateList.end(); itAUL++ )
  {
    if ( find( allAlignableDets.begin(), allAlignableDets.end(), *itAUL ) == allAlignableDets.end() )
    {
      allAlignableDets.push_back( *itAUL );
      additionalAlignableDets.push_back( *itAUL );
    }
  }

  //vector< Alignable* > currentAlignables = store->alignablesFromAlignableDets( currentAlignableDets );
  //vector< Alignable* > additionalAlignables = store->alignablesFromAlignableDets( additionalAlignableDets );
  //vector< Alignable* > allAlignables = store->alignablesFromAlignableDets( allAlignableDets );
  vector< Alignable* > currentAlignables = alignablesFromAlignableDets( currentAlignableDets, store );
  vector< Alignable* > additionalAlignables = alignablesFromAlignableDets( additionalAlignableDets, store );
  vector< Alignable* > allAlignables = alignablesFromAlignableDets( allAlignableDets, store );

  CompositeAlignmentParameters currentParameters = store->selectParameters( currentAlignableDets );
  AlgebraicSymMatrix currentAlignmentCov = currentParameters.covariance();

  CompositeAlignmentParameters additionalParameters = store->selectParameters( additionalAlignableDets );
  AlgebraicSymMatrix additionalAlignmentCov = additionalParameters.covariance();

  CompositeAlignmentParameters allParameters = store->selectParameters( allAlignableDets );
  AlgebraicVector allAlignmentParameters = allParameters.parameters();
  AlgebraicSymMatrix fullAlignmentCov = allParameters.covariance();
  AlgebraicMatrix mixedAlignmentCov = allParameters.covarianceSubset( additionalParameters.components(), currentParameters.components() );
  AlgebraicMatrix alignmentCovSubset = allParameters.covarianceSubset( allParameters.components(), currentParameters.components() );

  //CompositeAlignmentDerivativesExtractor extractor( currentAlignables, currentAlignableDets, ( *trajectory ).trajectoryStates() );
  //AlgebraicVector correctionTerm = extractor.correctionTerm();
  //AlgebraicMatrix alignmentDeriv = extractor.derivatives();
  AlgebraicVector correctionTerm = currentParameters.correctionTerm( ( *trajectory ).trajectoryStates(), currentAlignableDets );
  AlgebraicMatrix alignmentDeriv = currentParameters.derivatives( ( *trajectory ).trajectoryStates(), currentAlignableDets );
  AlgebraicSymMatrix alignmentCov = currentAlignmentCov.similarity( alignmentDeriv );

  AlgebraicVector allMeasurements = ( *trajectory ).measurements();
  AlgebraicSymMatrix measurementCov = ( *trajectory ).measurementErrors();
  AlgebraicVector referenceTrajectory = ( *trajectory ).trajectoryPositions();
  AlgebraicMatrix derivatives = ( *trajectory ).derivatives();

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

  AlgebraicMatrix gTimesDeriv = limitCov*alignmentDeriv;
  AlgebraicMatrix simMat = AlgebraicMatrix( nCRow, nCRow, 1 ) - covTimesDeriv*gTimesDeriv;
  AlgebraicSymMatrix updatedCurrentAlignmentCov = currentAlignmentCov.similarity( simMat ) + measurementCov.similarity( gainMatrix );

  AlgebraicMatrix mixedUpdateMat = simMat.T()*simMat.T() + measurementCov.similarity( gTimesDeriv.T() )*currentAlignmentCov;
  AlgebraicMatrix updatedMixedAlignmentCov = mixedAlignmentCov*mixedUpdateMat;

  AlgebraicSymMatrix additionalUpdateMat = misalignedCov.similarity( gTimesDeriv.T() ) - 2.*limitCov.similarity( alignmentDeriv.T() );
  AlgebraicSymMatrix updatedAdditionalAlignmentCov = additionalAlignmentCov + additionalUpdateMat.similarity( mixedAlignmentCov );

  AlgebraicSymMatrix updatedAlignmentCov( nCRow + nARow, 0 );

  for ( int nRow=0; nRow<nCRow; nRow++ )
  {
    for ( int nCol=0; nCol<nCRow; nCol++ ) updatedAlignmentCov[nRow][nCol] = updatedCurrentAlignmentCov[nRow][nCol];
  }

  for ( int nRow=0; nRow<nARow; nRow++ )
  {
    for ( int nCol=0; nCol<nCRow; nCol++ ) updatedAlignmentCov[nRow+nCRow][nCol] = updatedMixedAlignmentCov[nRow][nCol];
  }

  for ( int nRow=0; nRow<nARow; nRow++ )
  {
    for ( int nCol=0; nCol<nARow; nCol++ ) updatedAlignmentCov[nRow+nCRow][nCol+nCRow] = updatedAdditionalAlignmentCov[nRow][nCol];
  }

  // update in alignment-interface
  CompositeAlignmentParameters* updatedParameters;
  updatedParameters = allParameters.clone( updatedAlignmentParameters, updatedAlignmentCov );
  store->updateParameters( *updatedParameters );
  delete updatedParameters;

  // update user variables for debugging
  updateUserVariables( allAlignables );

  return;
}
