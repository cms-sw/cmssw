
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentInitialization.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"
#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include "CLHEP/Random/RandGauss.h"


KalmanAlignmentInitialization::KalmanAlignmentInitialization( const edm::ParameterSet& config ) : theConfiguration( config ) {}


KalmanAlignmentInitialization::~KalmanAlignmentInitialization( void ) {}


void KalmanAlignmentInitialization::initializeAlignmentParameters( AlignmentParameterStore* store )
{
  int updateGraph = theConfiguration.getUntrackedParameter< int >( "UpdateGraphs", 100 );

  double xFixedStartError = theConfiguration.getUntrackedParameter< double >( "XFixedStartError", 1e-10 );
  double yFixedStartError = theConfiguration.getUntrackedParameter< double >( "YFixedStartError", 1e-10 );
  double zFixedStartError = theConfiguration.getUntrackedParameter< double >( "ZFixedStartError", 1e-10 );
  double alphaFixedStartError = theConfiguration.getUntrackedParameter< double >( "AlphaFixedStartError", 1e-12 );
  double betaFixedStartError = theConfiguration.getUntrackedParameter< double >( "BetaFixedStartError", 1e-12 );
  double gammaFixedStartError = theConfiguration.getUntrackedParameter< double >( "GammaFixedStartError", 1e-12 );

  double xStartError = theConfiguration.getUntrackedParameter< double >( "XStartError", 4e-4 );
  double yStartError = theConfiguration.getUntrackedParameter< double >( "YStartError", 4e-4 );
  double zStartError = theConfiguration.getUntrackedParameter< double >( "ZStartError", 1e-2 );
  double alphaStartError = theConfiguration.getUntrackedParameter< double >( "AlphaStartError", 3e-5 );
  double betaStartError = theConfiguration.getUntrackedParameter< double >( "BetaStartError", 3e-5 );
  double gammaStartError = theConfiguration.getUntrackedParameter< double >( "GammaStartError", 3e-5 );

  bool applyXShifts =  theConfiguration.getUntrackedParameter< bool >( "ApplyXShifts", false );
  bool applyYShifts =  theConfiguration.getUntrackedParameter< bool >( "ApplyYShifts", false );
  bool applyZShifts =  theConfiguration.getUntrackedParameter< bool >( "ApplyZShifts", false );
  bool applyXRots =  theConfiguration.getUntrackedParameter< bool >( "ApplyXRotations", false );
  bool applyYRots =  theConfiguration.getUntrackedParameter< bool >( "ApplyYRotations", false );
  bool applyZRots =  theConfiguration.getUntrackedParameter< bool >( "ApplyZRotations", false );

  double sigmaXShift =  theConfiguration.getUntrackedParameter< double >( "SigmaXShifts", 1e-2 );
  double sigmaYShift =  theConfiguration.getUntrackedParameter< double >( "SigmaYShifts", 1e-2 );
  double sigmaZShift =  theConfiguration.getUntrackedParameter< double >( "SigmaZShifts", 5e-2 );
  double sigmaXRot =  theConfiguration.getUntrackedParameter< double >( "SigmaXRotations", 5e-3 );
  double sigmaYRot =  theConfiguration.getUntrackedParameter< double >( "SigmaYRotations", 5e-3 );
  double sigmaZRot =  theConfiguration.getUntrackedParameter< double >( "SigmaZRotations", 5e-3 );

  bool addPositionError = theConfiguration.getUntrackedParameter< bool >( "AddPositionError", true );

  bool applyCurl =  theConfiguration.getUntrackedParameter< bool >( "ApplyCurl", false );
  double curlConst =  theConfiguration.getUntrackedParameter< double >( "CurlConstant", 1e-6 );

  int seed  = theConfiguration.getUntrackedParameter< int >( "RandomSeed", 1726354 );
  HepRandom::createInstance();
  HepRandom::setTheSeed( seed );

  bool applyShifts = applyXShifts || applyYShifts || applyZShifts;
  bool applyRots = applyXRots || applyYRots || applyZRots;
  bool applyMisalignment = applyShifts || applyRots || applyCurl;

  AlgebraicVector startParameters( 6, 0 );
  AlgebraicSymMatrix startError( 6, 0 );
  AlgebraicSymMatrix fixedError( 6, 0 );

  if ( applyMisalignment )
  {
    startError[0][0] = xStartError;
    startError[1][1] = yStartError;
    startError[2][2] = zStartError;
    startError[3][3] = alphaStartError;
    startError[4][4] = betaStartError;
    startError[5][5] = gammaStartError;

    fixedError[0][0] = xFixedStartError;
    fixedError[1][1] = yFixedStartError;
    fixedError[2][2] = zFixedStartError;
    fixedError[3][3] = alphaFixedStartError;
    fixedError[4][4] = betaFixedStartError;
    fixedError[5][5] = gammaFixedStartError;
  }

  AlgebraicVector displacement( 3 );
  AlgebraicVector eulerAngles( 3 );

  TrackerAlignableId* alignableId = new TrackerAlignableId;

  vector< Alignable* > alignables = store->alignables();
  vector< Alignable* >::iterator itAlignable;

  for ( itAlignable = alignables.begin(); itAlignable != alignables.end(); itAlignable++ )
  {
    AlignmentParameters* alignmentParameters = ( *itAlignable )->alignmentParameters();
    KalmanAlignmentUserVariables* auv = new KalmanAlignmentUserVariables( *itAlignable, alignableId, updateGraph );

    pair< int, int > typeAndLayer = alignableId->typeAndLayerFromAlignable( *itAlignable );

    if ( abs( typeAndLayer.first ) <= 2 )
    {
      // fix alignables in the pixel detector
      alignmentParameters = alignmentParameters->clone( startParameters, fixedError );
    }
    else
    {
      alignmentParameters = alignmentParameters->clone( startParameters, startError );

      displacement[0] = applyXShifts ? sigmaXShift*RandGauss::shoot() : 0.;
      displacement[1] = applyYShifts ? sigmaZShift*RandGauss::shoot() : 0.;
      displacement[2] = applyZShifts ? sigmaYShift*RandGauss::shoot() : 0.;

      if ( applyShifts ) 
      {
	LocalVector localShift = LocalVector( displacement[0], displacement[1], displacement[2] );
	GlobalVector globalShift = ( *itAlignable )->surface().toGlobal( localShift );
	( *itAlignable )->move( globalShift );
      }

      eulerAngles[0] = applyXRots ? sigmaXRot*RandGauss::shoot() : 0;
      eulerAngles[1] = applyYRots ? sigmaYRot*RandGauss::shoot() : 0;
      eulerAngles[2] = applyZRots ? sigmaZRot*RandGauss::shoot() : 0;

      if ( applyRots )
      {
	AlignmentTransformations TkAT;
	Surface::RotationType localRotation = TkAT.rotationType( TkAT.rotMatrix3( eulerAngles ) );
	( *itAlignable )->rotateInLocalFrame( localRotation );
      }

      if ( applyCurl )
      {
	double radius = ( *itAlignable )->globalPosition().perp();
	( *itAlignable )->rotateAroundGlobalZ( curlConst*radius );
      }

      if ( addPositionError )
      {
	( *itAlignable )->addAlignmentPositionError( AlignmentPositionError( sigmaXShift, sigmaYShift, sigmaZShift ) );
      }

//       AlgebraicVector trueParameters( 6 );
//       trueParameters[0] = displacement[0];
//       trueParameters[1] = displacement[1];
//       trueParameters[2] = displacement[2];
//       trueParameters[3] = eulerAngles[0];
//       trueParameters[4] = eulerAngles[1];
//       trueParameters[5] = eulerAngles[2];
//       alignmentParameters = alignmentParameters->clone( trueParameters, AlgebraicSymMatrix( 6, 0 ) );
//       alignmentParameters = alignmentParameters->clone( startParameters, AlgebraicSymMatrix( 6, 0 ) );
    }

    alignmentParameters->setUserVariables( auv );
    ( *itAlignable )->setAlignmentParameters( alignmentParameters );
  }
}
