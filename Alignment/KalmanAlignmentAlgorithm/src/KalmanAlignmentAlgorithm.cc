
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentAlgorithm.h"

// includes for alignment
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

// includes for retracking
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/CurrentAlignmentKFUpdator.h"

// miscellaneous includes
#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "CLHEP/Random/RandGauss.h"
#include <fstream>

using namespace std;

KalmanAlignmentAlgorithm::KalmanAlignmentAlgorithm( const edm::ParameterSet& config ) :
  AlignmentAlgorithmBase( config ),
  theConfiguration( config ),
  theRefitterAlgo( config )
{}


KalmanAlignmentAlgorithm::~KalmanAlignmentAlgorithm( void ) {}


void KalmanAlignmentAlgorithm::initialize( const edm::EventSetup& setup, 
					   AlignableTracker* tracker, 
					   AlignableMuon* muon,
					   AlignmentParameterStore* store )
{
  theParameterStore = store;
  theNavigator = new AlignableNavigator( tracker->components() );

  initializeTrajectoryFitter( setup );

  initializeAlignmentParameters( setup );

  string identifier;
  edm::ParameterSet config;

  identifier = theConfiguration.getParameter< string >( "AlignmentUpdator" );
  config = theConfiguration.getParameter< edm::ParameterSet >( identifier );
  theAlignmentUpdator = KalmanAlignmentUpdatorPlugin::get()->create( identifier, config );

  identifier = theConfiguration.getParameter< string >( "MetricsUpdator" );
  config = theConfiguration.getParameter< edm::ParameterSet >( identifier );
  theMetricsUpdator = KalmanAlignmentMetricsUpdatorPlugin::get()->create( identifier, config );

  config = theConfiguration.getParameter< edm::ParameterSet >( "TrajectoryFactory" );
  identifier = config.getParameter< string >( "TrajectoryFactoryName" );
  theTrajectoryFactory = TrajectoryFactoryPlugin::get()->create( identifier, config );

  theRefitterDebugFlag = theConfiguration.getUntrackedParameter< bool >( "DebugRefitter", true );

  KalmanAlignmentDataCollector::configure( theConfiguration.getParameter< edm::ParameterSet >( "DataCollector" ) );
}


void KalmanAlignmentAlgorithm::terminate( void )
{
  cout << "[KalmanAlignmentAlgorithm::terminate] start ..." << endl;

  KalmanAlignmentDataCollector::write();

  TimingReport* timing = TimingReport::current();
  timing->dump( cout );  

  string timingLogFile = theConfiguration.getUntrackedParameter< string >( "TimingLogFile", "timing.log" );

  ofstream* output = new ofstream( timingLogFile.c_str() );
  timing->dump( *output );
  output->close();
  delete output;

  delete theNavigator;
  delete theTrajectoryRefitter;
  delete theTrajectoryFactory;
  delete theAlignmentUpdator;

  cout << "[KalmanAlignmentAlgorithm::terminate] ... done." << endl;
}


void KalmanAlignmentAlgorithm::run( const edm::EventSetup & setup,
				    const ConstTrajTrackPairCollection & tracks )
{
  static int iEvent = 1;
  if ( iEvent % 100 == 0 ) cout << "[KalmanAlignmentAlgorithm::run] Event Nr. " << iEvent << endl;
  iEvent++;

  // Run the refitter algorithm.
  ConstTrajTrackPairCollection refittedTracks = refitTracks( setup, tracks );

  // Produce reference trajectories.
  ReferenceTrajectoryCollection trajectories = theTrajectoryFactory->trajectories( setup, refittedTracks );
  ReferenceTrajectoryCollection::iterator itTrajectories = trajectories.begin();

  // Run the alignment algorithm.
  while( itTrajectories != trajectories.end() )
  {
    try
    {
      theAlignmentUpdator->process( *itTrajectories, theParameterStore, theNavigator, theMetricsUpdator );
    }
    catch( cms::Exception& exception )
    {
      terminate();
      throw exception;
    }

    KalmanAlignmentDataCollector::fillHistogram( "Trajectory_RecHits", (*itTrajectories)->recHits().size() );

    ++itTrajectories;
  }

  // Clean-Up.
  for ( ConstTrajTrackPairCollection::const_iterator it = refittedTracks.begin(); it != refittedTracks.end(); ++it )
  {
    delete (*it).first;
    delete (*it).second;
  }
  refittedTracks.clear();
}


AlignmentAlgorithmBase::ConstTrajTrackPairCollection
KalmanAlignmentAlgorithm::refitTracks( const edm::EventSetup& setup,
				       const ConstTrajTrackPairCollection& tracks )
{
  ConstTrajTrackPairCollection result;

  // Retrieve what we need from the EventSetup
  edm::ESHandle<TrackerGeometry> aGeometry;
  edm::ESHandle<MagneticField> aMagneticField;
  edm::ESHandle<TrajectoryFitter> aTrajectoryFitter;
  edm::ESHandle<Propagator> aPropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> aRecHitBuilder;

  getFromES( setup, aGeometry, aMagneticField, aTrajectoryFitter, aPropagator, aRecHitBuilder );

  for ( ConstTrajTrackPairCollection::const_iterator it = tracks.begin(); it != tracks.end(); ++it )
  {
    // Create a track collection containing only one track.
    reco::TrackCollection aTrackCollection;
    aTrackCollection.push_back( *(*it).second );

    AlgoProductCollection algoResult;
    theRefitterAlgo.runWithTrack( aGeometry.product(), aMagneticField.product(), aTrackCollection,
				  theTrajectoryRefitter, aPropagator.product(),
				  aRecHitBuilder.product(), algoResult );

    // The resulting collection contains either no or just one refitted trajectory/track-pair
    if ( !algoResult.empty() )
    {
      ConstTrajTrackPair aTrajTrackPair( algoResult.front().first, algoResult.front().second.first );
      result.push_back( aTrajTrackPair );

      if ( theRefitterDebugFlag )
      {
	float origPt = (*it).second->pt();
	float origEta = (*it).second->eta();
	float origPhi = (*it).second->phi();
	float origNormChi2 = (*it).second->normalizedChi2();
	float origDz = (*it).second->dz();

	KalmanAlignmentDataCollector::fillHistogram( "OrigTrack_Pt", 1e-2*origPt );
	KalmanAlignmentDataCollector::fillHistogram( "OrigTrack_Eta", origEta );
	KalmanAlignmentDataCollector::fillHistogram( "OrigTrack_Phi", origPhi );
	KalmanAlignmentDataCollector::fillHistogram( "OrigTrack_NormChi2", origNormChi2 );
	KalmanAlignmentDataCollector::fillHistogram( "OrigTrack_DZ", origDz );

	float refitPt = algoResult.front().second.first->pt();
	float refitEta = algoResult.front().second.first->eta();
	float refitPhi = algoResult.front().second.first->phi();
	float refitNormChi2 = algoResult.front().second.first->normalizedChi2();
	float refitDz = algoResult.front().second.first->dz();

	KalmanAlignmentDataCollector::fillHistogram( "RefitTrack_Pt", refitPt );
	KalmanAlignmentDataCollector::fillHistogram( "RefitTrack_Eta", refitEta );
	KalmanAlignmentDataCollector::fillHistogram( "RefitTrack_Phi", refitPhi );
	KalmanAlignmentDataCollector::fillHistogram( "RefitTrack_NormChi2", refitNormChi2 );
	KalmanAlignmentDataCollector::fillHistogram( "RefitTrack_DZ", refitDz );

	KalmanAlignmentDataCollector::fillHistogram( "Track_RelativeDeltaPt", ( refitPt - origPt )/origPt );
	KalmanAlignmentDataCollector::fillHistogram( "Track_DeltaEta", refitEta - origEta );
	KalmanAlignmentDataCollector::fillHistogram( "Track_DeltaPhi", refitPhi - origPhi );
	KalmanAlignmentDataCollector::fillHistogram( "Track_DeltaNormChi2", refitNormChi2 - origNormChi2 );
      }
    }
  }

  return result;
}


void KalmanAlignmentAlgorithm::initializeTrajectoryFitter( const edm::EventSetup& setup )
{
  setConf( theConfiguration );

  edm::ESHandle< TrajectoryFitter > defaultTrajectoryFitter;
  string fitterName = theConfiguration.getParameter< string >( "Fitter" );
  setup.get< TrackingComponentsRecord >().get( fitterName, defaultTrajectoryFitter );

  const KFFittingSmoother* fittingSmoother = dynamic_cast< const KFFittingSmoother* >( defaultTrajectoryFitter.product() );
  if ( fittingSmoother )
  {
    TrajectoryFitter* newFitter = 0;
    TrajectorySmoother* newSmoother = 0;
    CurrentAlignmentKFUpdator* newUpdator = new CurrentAlignmentKFUpdator( theNavigator );

    const KFTrajectoryFitter* aKFFitter = dynamic_cast< const KFTrajectoryFitter* >( fittingSmoother->fitter() );
    if ( aKFFitter )
    {
      newFitter = new KFTrajectoryFitter( aKFFitter->propagator(), newUpdator, aKFFitter->estimator() );
    }

    const KFTrajectorySmoother* aKFSmoother = dynamic_cast< const KFTrajectorySmoother* >( fittingSmoother->smoother() );
    if ( aKFSmoother )
    {
      newSmoother = new KFTrajectorySmoother( aKFSmoother->propagator(), newUpdator, aKFSmoother->estimator() );
    }

    if ( newFitter && newSmoother )
    {
      theTrajectoryRefitter = new KFFittingSmoother( *newFitter, *newSmoother );
      delete newFitter;
      delete newSmoother;
    }

    delete newUpdator;
  }
}


void KalmanAlignmentAlgorithm::initializeAlignmentParameters( const edm::EventSetup& setup )
{
  const edm::ParameterSet initConfig = theConfiguration.getParameter< edm::ParameterSet >( "Initialization" );

  int updateGraph = initConfig.getUntrackedParameter< int >( "UpdateGraphs", 100 );

  bool applyXShifts =  initConfig.getUntrackedParameter< bool >( "ApplyXShifts", false );
  bool applyYShifts =  initConfig.getUntrackedParameter< bool >( "ApplyYShifts", false );
  bool applyZShifts =  initConfig.getUntrackedParameter< bool >( "ApplyZShifts", false );
  bool applyXRots =  initConfig.getUntrackedParameter< bool >( "ApplyXRotations", false );
  bool applyYRots =  initConfig.getUntrackedParameter< bool >( "ApplyYRotations", false );
  bool applyZRots =  initConfig.getUntrackedParameter< bool >( "ApplyZRotations", false );

  double sigmaXShift =  initConfig.getUntrackedParameter< double >( "SigmaXShifts", 1e-2 );
  double sigmaYShift =  initConfig.getUntrackedParameter< double >( "SigmaYShifts", 1e-2 );
  double sigmaZShift =  initConfig.getUntrackedParameter< double >( "SigmaZShifts", 5e-2 );
  double sigmaXRot =  initConfig.getUntrackedParameter< double >( "SigmaXRotations", 5e-3 );
  double sigmaYRot =  initConfig.getUntrackedParameter< double >( "SigmaYRotations", 5e-3 );
  double sigmaZRot =  initConfig.getUntrackedParameter< double >( "SigmaZRotations", 5e-3 );

  bool addPositionError = initConfig.getUntrackedParameter< bool >( "AddPositionError", true );

  bool applyCurl =  initConfig.getUntrackedParameter< bool >( "ApplyCurl", false );
  double curlConst =  initConfig.getUntrackedParameter< double >( "CurlConstant", 1e-6 );

  int seed  = initConfig.getUntrackedParameter< int >( "RandomSeed", 1726354 );
  HepRandom::createInstance();
  HepRandom::setTheSeed( seed );

  bool applyShifts = applyXShifts || applyYShifts || applyZShifts;
  bool applyRots = applyXRots || applyYRots || applyZRots;
  //bool applyMisalignment = applyShifts || applyRots || applyCurl;

  AlgebraicVector startParameters( 6, 0 );
  AlgebraicSymMatrix startError( 6, 0 );
  AlgebraicSymMatrix fixedError( 6, 0 );

  fixedError[0][0] = initConfig.getUntrackedParameter< double >( "XFixedStartError", 1e-10 );
  fixedError[1][1] = initConfig.getUntrackedParameter< double >( "YFixedStartError", 1e-10 );
  fixedError[2][2] = initConfig.getUntrackedParameter< double >( "ZFixedStartError", 1e-10 );
  fixedError[3][3] = initConfig.getUntrackedParameter< double >( "AlphaFixedStartError", 1e-12 );
  fixedError[4][4] = initConfig.getUntrackedParameter< double >( "BetaFixedStartError", 1e-12 );
  fixedError[5][5] = initConfig.getUntrackedParameter< double >( "GammaFixedStartError", 1e-12 );

  startError[0][0] = initConfig.getUntrackedParameter< double >( "XStartError", 4e-4 );
  startError[1][1] = initConfig.getUntrackedParameter< double >( "YStartError", 4e-4 );
  startError[2][2] = initConfig.getUntrackedParameter< double >( "ZStartError", 4e-4 );
  startError[3][3] = initConfig.getUntrackedParameter< double >( "AlphaStartError", 3e-5 );
  startError[4][4] = initConfig.getUntrackedParameter< double >( "BetaStartError", 3e-5 );
  startError[5][5] = initConfig.getUntrackedParameter< double >( "GammaStartError", 3e-5 );


  TrackerAlignableId* alignableId = new TrackerAlignableId;

  vector< Alignable* > alignables = theParameterStore->alignables();
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

      double shiftX = applyXShifts ? sigmaXShift*RandGauss::shoot() : 0.;
      double shiftY = applyYShifts ? sigmaZShift*RandGauss::shoot() : 0.;
      double shiftZ = applyZShifts ? sigmaYShift*RandGauss::shoot() : 0.;

      if ( applyShifts ) 
      {
	align::LocalVector localShift( shiftX, shiftY, shiftZ );
	align::GlobalVector globalShift = ( *itAlignable )->surface().toGlobal( localShift );
	( *itAlignable )->move( globalShift );
      }

      align::EulerAngles eulerAngles( 3 );

      eulerAngles[0] = applyXRots ? sigmaXRot*RandGauss::shoot() : 0.;
      eulerAngles[1] = applyYRots ? sigmaYRot*RandGauss::shoot() : 0.;
      eulerAngles[2] = applyZRots ? sigmaZRot*RandGauss::shoot() : 0.;

      if ( applyRots )
      {
	align::RotationType localRotation = align::toMatrix( eulerAngles );
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

