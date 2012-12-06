
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayEstimator.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayDerivatives.h"
#include "FWCore/Utilities/interface/isFinite.h"
//#include "DataFormats/CLHEP/interface/Migration.h"

TwoBodyDecayEstimator::TwoBodyDecayEstimator( const edm::ParameterSet & config )
  :theNdf(0)
{
  const edm::ParameterSet & estimatorConfig = config.getParameter< edm::ParameterSet >( "EstimatorParameters" );

  theRobustificationConstant = estimatorConfig.getUntrackedParameter< double >( "RobustificationConstant", 1.0 );
  theMaxIterDiff = estimatorConfig.getUntrackedParameter< double >( "MaxIterationDifference", 1e-2 );
  theMaxIterations = estimatorConfig.getUntrackedParameter< int >( "MaxIterations", 100 );
  theUseInvariantMass = estimatorConfig.getUntrackedParameter< bool >( "UseInvariantMass", true );
}


TwoBodyDecay TwoBodyDecayEstimator::estimate( const std::vector< RefCountedLinearizedTrackState > & linTracks,
					      const TwoBodyDecayParameters & linearizationPoint,
					      const TwoBodyDecayVirtualMeasurement & vm ) const
{
  if ( linTracks.size() != 2 )
  {
    edm::LogInfo( "Alignment" ) << "@SUB=TwoBodyDecayEstimator::estimate"
				<< "Need 2 linearized tracks, got " << linTracks.size() << ".\n";
    return TwoBodyDecay();
  }

  AlgebraicVector vecM;
  AlgebraicSymMatrix matG;
  AlgebraicMatrix matA;

  bool check = constructMatrices( linTracks, linearizationPoint, vm, vecM, matG, matA );
  if ( !check ) return TwoBodyDecay();

  AlgebraicSymMatrix matGPrime;
  AlgebraicSymMatrix invAtGPrimeA;
  AlgebraicVector vecEstimate;
  AlgebraicVector res;

  int nIterations = 0;
  bool stopIteration = false;

  // initialization - values are never used
  int checkInversion = 0;
  double chi2 = 0.;
  double oldChi2 = 0.;
  bool isValid = true;

  while( !stopIteration )
  {
    matGPrime = matG;

    // compute weights
    if ( nIterations > 0 )
    {
      for ( int i = 0; i < 10; i++ )
      {
	double sigma = 1./sqrt( matG[i][i] );
	double sigmaTimesR = sigma*theRobustificationConstant;
	double absRes = fabs( res[i] ); 
	if (  absRes > sigmaTimesR ) matGPrime[i][i] *= sigmaTimesR/absRes;
      }
    }

    // make LS-fit
    invAtGPrimeA = ( matGPrime.similarityT(matA) ).inverse( checkInversion );
    if ( checkInversion != 0 )
    {
      LogDebug( "Alignment" ) << "@SUB=TwoBodyDecayEstimator::estimate"
			      << "Matrix At*G'*A not invertible (in iteration " << nIterations
			      << ", ifail = " << checkInversion << ").\n";
      isValid = false;
      break;
    }
    vecEstimate = invAtGPrimeA*matA.T()*matGPrime*vecM;
    res = matA*vecEstimate - vecM;
    chi2 = dot( res, matGPrime*res );

    if ( ( nIterations > 0 ) && ( fabs( chi2 - oldChi2 ) < theMaxIterDiff ) ) stopIteration = true;
    if ( nIterations == theMaxIterations ) stopIteration = true;

    oldChi2 = chi2;
    nIterations++;
  }

  if ( isValid )
  {
    AlgebraicSymMatrix pullsCov = matGPrime.inverse( checkInversion ) - invAtGPrimeA.similarity( matA );
    thePulls = AlgebraicVector( matG.num_col(), 0 );
    for ( int i = 0; i < pullsCov.num_col(); i++ ) thePulls[i] = res[i]/sqrt( pullsCov[i][i] );
  }

  theNdf = matA.num_row() - matA.num_col();

  return TwoBodyDecay( TwoBodyDecayParameters( vecEstimate, invAtGPrimeA ), chi2, isValid, vm );
}


bool TwoBodyDecayEstimator::constructMatrices( const std::vector< RefCountedLinearizedTrackState > & linTracks,
					       const TwoBodyDecayParameters & linearizationPoint,
					       const TwoBodyDecayVirtualMeasurement & vm,
					       AlgebraicVector & vecM, AlgebraicSymMatrix & matG, AlgebraicMatrix & matA ) const
{

  PerigeeLinearizedTrackState* linTrack1 = dynamic_cast<PerigeeLinearizedTrackState*>( linTracks[0].get() );
  PerigeeLinearizedTrackState* linTrack2 = dynamic_cast<PerigeeLinearizedTrackState*>( linTracks[1].get() );

  if (!linTrack1 || !linTrack2) return false;

  AlgebraicVector trackParam1 = asHepVector( linTrack1->predictedStateParameters() );
  AlgebraicVector trackParam2 = asHepVector( linTrack2->predictedStateParameters() );

  if ( checkValues( trackParam1 ) || checkValues( trackParam2 ) || checkValues( linearizationPoint.parameters() ) ) return false;

  AlgebraicVector vecLinParam = linearizationPoint.sub( TwoBodyDecayParameters::px,
							TwoBodyDecayParameters::mass );

  double zMagField = linTrack1->track().field()->inInverseGeV( linTrack1->linearizationPoint() ).z();

  int checkInversion = 0;

  TwoBodyDecayDerivatives tpeDerivatives( linearizationPoint[TwoBodyDecayParameters::mass], vm.secondaryMass() );
  std::pair< AlgebraicMatrix, AlgebraicMatrix > derivatives = tpeDerivatives.derivatives( linearizationPoint );

  TwoBodyDecayModel decayModel( linearizationPoint[TwoBodyDecayParameters::mass], vm.secondaryMass() );
  std::pair< AlgebraicVector, AlgebraicVector > linCartMomenta = decayModel.cartesianSecondaryMomenta( linearizationPoint );

  // first track
  AlgebraicMatrix matA1 = asHepMatrix( linTrack1->positionJacobian() );
  AlgebraicMatrix matB1 = asHepMatrix( linTrack1->momentumJacobian() );
  AlgebraicVector vecC1 = asHepVector( linTrack1->constantTerm() );

  AlgebraicVector curvMomentum1 = asHepVector( linTrack1->predictedStateMomentumParameters() );
  AlgebraicMatrix curv2cart1 = decayModel.curvilinearToCartesianJacobian( curvMomentum1, zMagField );

  AlgebraicVector cartMomentum1 = decayModel.convertCurvilinearToCartesian( curvMomentum1, zMagField );
  vecC1 += matB1*( curvMomentum1 - curv2cart1*cartMomentum1 );
  matB1 = matB1*curv2cart1;

  AlgebraicMatrix matF1 = derivatives.first;
  AlgebraicVector vecD1 = linCartMomenta.first - matF1*vecLinParam;
  AlgebraicVector vecM1 = trackParam1 - vecC1 - matB1*vecD1;
  AlgebraicSymMatrix covM1 = asHepMatrix( linTrack1->predictedStateError() );


  AlgebraicSymMatrix matG1 = covM1.inverse( checkInversion );
  if ( checkInversion != 0 )
  {
    LogDebug( "Alignment" ) << "@SUB=TwoBodyDecayEstimator::constructMatrices"
			    << "Matrix covM1 not invertible.";
    return false;
  }

  // diagonalize for robustification
   AlgebraicMatrix matU1 = diagonalize( &matG1 ).T();

  // second track
  AlgebraicMatrix matA2 = asHepMatrix( linTrack2->positionJacobian() );
  AlgebraicMatrix matB2 = asHepMatrix( linTrack2->momentumJacobian() );
  AlgebraicVector vecC2 = asHepVector( linTrack2->constantTerm() );

  AlgebraicVector curvMomentum2 = asHepVector( linTrack2->predictedStateMomentumParameters() );
  AlgebraicMatrix curv2cart2 = decayModel.curvilinearToCartesianJacobian( curvMomentum2, zMagField );

  AlgebraicVector cartMomentum2 = decayModel.convertCurvilinearToCartesian( curvMomentum2, zMagField );
  vecC2 += matB2*( curvMomentum2 - curv2cart2*cartMomentum2 );
  matB2 = matB2*curv2cart2;

  AlgebraicMatrix matF2 = derivatives.second;
  AlgebraicVector vecD2 = linCartMomenta.second - matF2*vecLinParam;
  AlgebraicVector vecM2 = trackParam2 - vecC2 - matB2*vecD2;
  AlgebraicSymMatrix covM2 = asHepMatrix( linTrack2->predictedStateError() );

  AlgebraicSymMatrix matG2 = covM2.inverse( checkInversion );
  if ( checkInversion != 0 )
  {
    LogDebug( "Alignment" ) << "@SUB=TwoBodyDecayEstimator::constructMatrices"
			    << "Matrix covM2 not invertible.";
    return false;
  }

  // diagonalize for robustification
  AlgebraicMatrix matU2 = diagonalize( &matG2 ).T();

  // combine first and second track
  vecM = AlgebraicVector( 14, 0 );
  vecM.sub( 1, matU1*vecM1 );
  vecM.sub( 6, matU2*vecM2 );
  // virtual measurement of the primary mass
  vecM( 11 ) = vm.primaryMass();
  // virtual measurement of the beam spot
  vecM.sub( 12, vm.beamSpotPosition() );

  // full weight matrix
  matG = AlgebraicSymMatrix( 14, 0 );
  matG.sub( 1, matG1 );
  matG.sub( 6, matG2 );
  // virtual measurement error of the primary mass
  matG[10][10] = 1./( vm.primaryWidth()*vm.primaryWidth() );
  // virtual measurement error of the beam spot
  matG.sub( 12, vm.beamSpotError().inverse( checkInversion ) );

  // full derivative matrix
  matA = AlgebraicMatrix( 14, 9, 0 );
  matA.sub( 1, 1, matU1*matA1 );
  matA.sub( 6, 1, matU2*matA2 );
  matA.sub( 1, 4, matU1*matB1*matF1 );
  matA.sub( 6, 4, matU2*matB2*matF2 );
  matA( 11, 9 ) = 1.;//mass
  matA( 12, 1 ) = 1.;//vx
  matA( 13, 2 ) = 1.;//vy
  matA( 14, 3 ) = 1.;//vz

  return true;
}


bool TwoBodyDecayEstimator::checkValues( const AlgebraicVector & vec ) const
{
  bool isNotFinite = false;

  for ( int i = 0; i < vec.num_col(); ++i )
    isNotFinite |= edm::isNotFinite( vec[i] );

  return isNotFinite;
}
