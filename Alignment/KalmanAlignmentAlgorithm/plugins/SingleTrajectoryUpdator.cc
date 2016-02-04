
//#include "Alignment/KalmanAlignmentAlgorithm/plugins/SingleTrajectoryUpdator.h"
#include "SingleTrajectoryUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"

#include "Alignment/CommonAlignmentParametrization/interface/CompositeAlignmentDerivativesExtractor.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

#include <algorithm>


using namespace std;


SingleTrajectoryUpdator::SingleTrajectoryUpdator( const edm::ParameterSet & config ) :
  KalmanAlignmentUpdator( config )
{
  theMinNumberOfHits = config.getParameter< unsigned int >( "MinNumberOfHits" );
  theExtraWeight = config.getParameter< double >( "ExtraWeight" );
  theExternalPredictionWeight = config.getParameter< double >( "ExternalPredictionWeight" );
  theCovCheckFlag = config.getParameter< bool >( "CheckCovariance" );

  theNumberOfPreAlignmentEvts = config.getParameter< unsigned int >( "NumberOfPreAlignmentEvts" );
  theNumberOfProcessedEvts = 0;

  std::cout << "[SingleTrajectoryUpdator] Use " << theNumberOfPreAlignmentEvts << "events for pre-alignment" << std::endl;
}


SingleTrajectoryUpdator::~SingleTrajectoryUpdator( void ) {}


void SingleTrajectoryUpdator::process( const ReferenceTrajectoryPtr & trajectory,
				       AlignmentParameterStore* store,
				       AlignableNavigator* navigator,
				       KalmanAlignmentMetricsUpdator* metrics,
				       const MagneticField* magField )
{
  if ( !( *trajectory ).isValid() ) return;

//   std::cout << "[SingleTrajectoryUpdator::process] START" << std::endl;

  vector< AlignableDetOrUnitPtr > currentAlignableDets = navigator->alignablesFromHits( ( *trajectory ).recHits() );
  vector< Alignable* > currentAlignables = alignablesFromAlignableDets( currentAlignableDets, store );

  if ( nDifferentAlignables( currentAlignables ) < 2 ) return;
  if ( currentAlignables.size() < theMinNumberOfHits ) return;

  ++theNumberOfProcessedEvts;
  bool includeCorrelations = ( theNumberOfPreAlignmentEvts < theNumberOfProcessedEvts );
  
  metrics->update( currentAlignables );

  vector< Alignable* > additionalAlignables;
  if ( includeCorrelations ) additionalAlignables = metrics->additionalAlignables( currentAlignables );

  vector< Alignable* > allAlignables;
  allAlignables.reserve( currentAlignables.size() + additionalAlignables.size() );
  allAlignables.insert( allAlignables.end(), currentAlignables.begin(), currentAlignables.end() );
  allAlignables.insert( allAlignables.end(), additionalAlignables.begin(), additionalAlignables.end() );

  CompositeAlignmentParameters alignmentParameters = store->selectParameters( allAlignables );

  const AlgebraicVector& allAlignmentParameters = alignmentParameters.parameters();
  AlgebraicSymMatrix currentAlignmentCov = alignmentParameters.covarianceSubset( currentAlignables );
  AlgebraicSymMatrix additionalAlignmentCov = alignmentParameters.covarianceSubset( additionalAlignables );
  AlgebraicMatrix mixedAlignmentCov = alignmentParameters.covarianceSubset( additionalAlignables, currentAlignables );
  AlgebraicMatrix alignmentCovSubset = alignmentParameters.covarianceSubset( alignmentParameters.components(), currentAlignables );

  CompositeAlignmentDerivativesExtractor extractor( currentAlignables, currentAlignableDets, trajectory->trajectoryStates() );
  AlgebraicVector correctionTerm = extractor.correctionTerm();
  AlgebraicMatrix alignmentDeriv = extractor.derivatives();
  AlgebraicSymMatrix alignmentCov = currentAlignmentCov.similarity( alignmentDeriv );

  AlgebraicVector allMeasurements = trajectory->measurements();
  AlgebraicSymMatrix measurementCov = trajectory->measurementErrors();
  AlgebraicVector referenceTrajectory = trajectory->trajectoryPositions();
  AlgebraicMatrix derivatives = trajectory->derivatives();

  measurementCov += theExtraWeight*AlgebraicSymMatrix( measurementCov.num_row(), 1 );

  AlgebraicSymMatrix misalignedCov = measurementCov + alignmentCov;

  int checkInversion = 0;

  AlgebraicSymMatrix weightMatrix;
  AlgebraicVector residuals;

  if ( trajectory->parameterErrorsAvailable() ) // Make an update using an external prediction for the track parameters.
  {
    const AlgebraicSymMatrix& externalParamCov = trajectory->parameterErrors();
    AlgebraicSymMatrix externalTrackCov = theExternalPredictionWeight*externalParamCov.similarity( derivatives );
    AlgebraicSymMatrix fullCov = misalignedCov + externalTrackCov;
    measurementCov += externalTrackCov;

    weightMatrix = fullCov.inverse( checkInversion );
    if ( checkInversion != 0 )
    {
      cout << "[KalmanAlignment] WARNING: 'AlgebraicSymMatrix fullCov' not invertible." << endl;
      return;
    }

    //const AlgebraicVector& trackParameters = trajectory->parameters();
    //const AlgebraicVector& externalTrackParameters = trajectory->externalPrediction();
    //AlgebraicVector trackCorrectionTerm = derivatives*( externalTrackParameters - trackParameters );
    residuals = allMeasurements - referenceTrajectory - correctionTerm;// - trackCorrectionTerm;
  }
  else // No external prediction for the track parameters available --> give the track parameters weight 0.
  {
    AlgebraicSymMatrix invMisalignedCov = misalignedCov.inverse( checkInversion );
    if ( checkInversion != 0 )
    {
      cout << "[KalmanAlignment] WARNING: 'AlgebraicSymMatrix misalignedCov' not invertible." << endl;
      return;
    }
    AlgebraicSymMatrix weightMatrix1 = ( invMisalignedCov.similarityT( derivatives ) ).inverse( checkInversion );
    if ( checkInversion != 0 )
    {
      cout << "[KalmanAlignment] WARNING: 'AlgebraicSymMatrix weightMatrix1' not computed." << endl;
      return;
    }
    AlgebraicSymMatrix weightMatrix2 = weightMatrix1.similarity( invMisalignedCov*derivatives );

    weightMatrix = invMisalignedCov - weightMatrix2;
    residuals = allMeasurements - referenceTrajectory - correctionTerm;
  }

//   AlgebraicVector deltaR = allMeasurements - referenceTrajectory;
//   for ( int i = 0; i < deltaR.num_row()/2; ++i )
//     KalmanAlignmentDataCollector::fillHistogram( "DeltaR_", i, deltaR[2*i] );
//   return;

  AlgebraicMatrix fullCovTimesDeriv = alignmentCovSubset*alignmentDeriv.T();
  AlgebraicMatrix fullGainMatrix = fullCovTimesDeriv*weightMatrix;
  AlgebraicMatrix covTimesDeriv = currentAlignmentCov*alignmentDeriv.T();
  AlgebraicMatrix gainMatrix = covTimesDeriv*weightMatrix;

  // make updates for the kalman-filter
  // update of parameters
  AlgebraicVector updatedAlignmentParameters = allAlignmentParameters + fullGainMatrix*residuals;

  // update of covariance
  int nCRow = currentAlignmentCov.num_row();
  int nARow = additionalAlignmentCov.num_row();

  AlgebraicSymMatrix updatedAlignmentCov( nCRow + nARow );

  AlgebraicMatrix gTimesDeriv = weightMatrix*alignmentDeriv;
  AlgebraicMatrix simMat = AlgebraicMatrix( nCRow, nCRow, 1 ) - covTimesDeriv*gTimesDeriv;
  AlgebraicSymMatrix updatedCurrentAlignmentCov = currentAlignmentCov.similarity( simMat ) + measurementCov.similarity( gainMatrix );

  AlgebraicMatrix mixedUpdateMat = simMat.T()*simMat.T() + measurementCov.similarity( gTimesDeriv.T() )*currentAlignmentCov;
  AlgebraicMatrix updatedMixedAlignmentCov = mixedAlignmentCov*mixedUpdateMat;

  AlgebraicSymMatrix additionalUpdateMat = misalignedCov.similarity( gTimesDeriv.T() ) - 2.*weightMatrix.similarity( alignmentDeriv.T() );
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


  // update in alignment-interface
  CompositeAlignmentParameters* updatedParameters;
  updatedParameters = alignmentParameters.clone( updatedAlignmentParameters, updatedAlignmentCov );

  if ( !checkCovariance( updatedAlignmentCov ) )
  {
    if ( includeCorrelations ) throw cms::Exception( "BadCovariance" );

    delete updatedParameters;
    return;
  }

  store->updateParameters( *updatedParameters, includeCorrelations );
  delete updatedParameters;


  // update user variables for debugging
  //updateUserVariables( alignmentParameters.components() );

  //std::cout << "update user variables now" << std::endl;
  updateUserVariables( currentAlignables );
  //std::cout << "done." << std::endl;

  static int i = 0;
  if ( i%100 == 0 ) KalmanAlignmentDataCollector::fillGraph( "correlation_entries", i, store->numCorrelations() );
  ++i;

  //std::cout << "[SingleTrajectoryUpdator::process] DONE" << std::endl;

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


DEFINE_EDM_PLUGIN( KalmanAlignmentUpdatorPlugin, SingleTrajectoryUpdator, "SingleTrajectoryUpdator" );
