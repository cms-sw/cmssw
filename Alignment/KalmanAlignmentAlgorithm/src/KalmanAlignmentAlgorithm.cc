
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentAlgorithm.h"

// includes for alignment
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentIORoot.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentParameterSelector.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryPlugin.h"

#include "Alignment/TrackerAlignment/interface/TrackerAlignableId.h"
#include "Alignment/TrackerAlignment/interface/AlignableTracker.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

// miscellaneous includes
#include "FWCore/Utilities/interface/Exception.h"
#include "Utilities/Timing/interface/TimingReport.h"
#include "CLHEP/Random/RandGauss.h"
#include <fstream>

using namespace std;


KalmanAlignmentAlgorithm::KalmanAlignmentAlgorithm( const edm::ParameterSet& config ) :
  AlignmentAlgorithmBase( config ),
  theConfiguration( config )
{}


KalmanAlignmentAlgorithm::~KalmanAlignmentAlgorithm( void ) {}


void KalmanAlignmentAlgorithm::initialize( const edm::EventSetup& setup, 
					   AlignableTracker* tracker, 
					   AlignableMuon* muon,
					   AlignmentParameterStore* store )
{
  theParameterStore = store;
  theNavigator = new AlignableNavigator( tracker->components() );
  theSelector = new AlignmentParameterSelector( tracker );

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

  config = theConfiguration.getParameter< edm::ParameterSet >( "TrackRefitter" );
  theRefitter = new KalmanAlignmentTrackRefitter( config );
  theRefitter->initialize( setup, theNavigator );

  KalmanAlignmentDataCollector::configure( theConfiguration.getParameter< edm::ParameterSet >( "DataCollector" ) );
}


void KalmanAlignmentAlgorithm::terminate( void )
{
  cout << "[KalmanAlignmentAlgorithm::terminate] start ..." << endl;

  vector< Alignable* > allAlignables = theMetricsUpdator->alignables();
  cout << "[KalmanAlignmentAlgorithm::terminate] The metrics updator holds " << allAlignables.size() << " alignables" << endl;

  //
  // Determine for which alignables the (debug) data shall be written to file.
  //
  vector< Alignable* > alignablesToWrite;
  for ( vector< Alignable* >::iterator it = allAlignables.begin(); it != allAlignables.end(); ++it )
  {
    AlignmentParameters* alignmentParameters = ( *it )->alignmentParameters();

    if ( alignmentParameters != 0 )
    {
      KalmanAlignmentUserVariables* userVariables =
	dynamic_cast< KalmanAlignmentUserVariables* >( alignmentParameters->userVariables() );

      if ( userVariables != 0 && userVariables->numberOfUpdates() > 0 )
      {
	userVariables->update( true );
	userVariables->histogramParameters( "KalmanAlignmentAlgorithm" );
	alignablesToWrite.push_back( *it );
      }
    }
  }

  //
  // Write alignment parameters and errors to file.
  //
  if ( theConfiguration.getUntrackedParameter< bool >( "WriteAlignmentParameters", false ) )
  {
    AlignmentIORoot alignmentIO;
    int ierr = 0;
    string output = theConfiguration.getParameter< string >( "OutputFile" );

    cout << "Write data for " << alignablesToWrite.size() << " alignables ..." << endl;

    // Write output to "iteration 1", ...
    alignmentIO.writeAlignmentParameters( alignablesToWrite, output.c_str(), 1, false, ierr );
    // ... or, if "iteration 1" already exists, write it to "highest iteration + 1".
    if ( ierr == -1 ) alignmentIO.writeAlignmentParameters( alignablesToWrite, output.c_str(), -1, false, ierr );
  }

  //
  // Write debug data to file.
  //
  KalmanAlignmentDataCollector::write();

  //
  // Write timing to file.
  //
  TimingReport* timing = TimingReport::current();
  timing->dump( cout );  

  string timingLogFile = theConfiguration.getUntrackedParameter< string >( "TimingLogFile", "timing.log" );

  ofstream* output = new ofstream( timingLogFile.c_str() );
  timing->dump( *output );
  output->close();
  delete output;

  delete theNavigator;
  delete theTrajectoryFactory;
  delete theAlignmentUpdator;

  cout << "[KalmanAlignmentAlgorithm::terminate] ... done." << endl;
}


void KalmanAlignmentAlgorithm::run( const edm::EventSetup & setup,
				    const ConstTrajTrackPairCollection & tracks )
{
  static int iEvent = 1;
  if ( iEvent % 500 == 0 )  cout << "[KalmanAlignmentAlgorithm::run] Event Nr. " << iEvent << endl;
  iEvent++;

  //
  // Run the refitter algorithm.
  //
  TrackletCollection refittedTracklets = theRefitter->refitTracks( setup, tracks );

  //
  // Construct reference trajectories from refitter output.
  //
  ConstTrajTrackPairCollection tracklets;
  ExternalPredictionCollection external;

  TrackletCollection::iterator itTracklet;
  for ( itTracklet = refittedTracklets.begin(); itTracklet != refittedTracklets.end(); ++itTracklet )
  {
    tracklets.push_back( (*itTracklet)->trajTrackPair() );
    external.push_back( (*itTracklet)->externalPrediction() );
  }

  ReferenceTrajectoryCollection trajectories = theTrajectoryFactory->trajectories( setup, tracklets, external );

  //
  // Run the alignment algorithm.
  //
  ReferenceTrajectoryCollection::iterator itTrajectories;
  for ( itTrajectories = trajectories.begin(); itTrajectories != trajectories.end(); ++itTrajectories )
    theAlignmentUpdator->process( *itTrajectories, theParameterStore, theNavigator, theMetricsUpdator );
}


void KalmanAlignmentAlgorithm::initializeAlignmentParameters( const edm::EventSetup& setup )
{
  TrackerAlignableId* alignableId = new TrackerAlignableId;

  const edm::ParameterSet initConfig = theConfiguration.getParameter< edm::ParameterSet >( "Initialization" );

  int updateGraph = initConfig.getUntrackedParameter< int >( "UpdateGraphs", 100 );
  bool addPositionError = initConfig.getUntrackedParameter< bool >( "AddPositionError", false );
  bool applyRandomStartValues = initConfig.getUntrackedParameter< bool >( "ApplyRandomStartValues", false );
  if ( applyRandomStartValues ) cout << "ADDING RANDOM START VALUES!!!" << endl;

  AlgebraicVector startParameters( 6, 0 );
  AlgebraicSymMatrix startError( 6, 0 );
  AlgebraicVector randSig( 6, 0 );

  vector< string > initSelection = initConfig.getParameter< vector<string> >( "InitializationSelector" );
  vector< string >::iterator itInitSel;
  for ( itInitSel = initSelection.begin(); itInitSel != initSelection.end(); ++itInitSel )
  {
    const edm::ParameterSet config = initConfig.getParameter< edm::ParameterSet >( *itInitSel );

    //
    // Retrieve alignables that shall be initialized.
    //
    vector< char > dummyParamSelector( 6, '0' );
    vector< string > alignableSelector = config.getParameter< vector<string> >( "AlignableSelection" );
    vector< string >::const_iterator itAliSel;
    for ( itAliSel = alignableSelector.begin(); itAliSel != alignableSelector.end(); ++itAliSel )
    {
      theSelector->addSelection( *itAliSel, dummyParamSelector );
      cout << "[" << *itInitSel << "] add selection: " << *itAliSel << endl;
    }
    vector< Alignable* > alignables = theSelector->selectedAlignables();

    //
    // Retrieve alignment information from file.
    //
    bool readParam = config.getUntrackedParameter< bool >( "ReadParametersFromFile", false );
    bool readCovar = config.getUntrackedParameter< bool >( "ReadCovarianceFromFile", false );
    bool applyParam = config.getUntrackedParameter< bool >( "ApplyParametersFromFile", false );
    bool applyCovar = config.getUntrackedParameter< bool >( "ApplyErrorFromFile", false );

    map< Alignable*, vector< AlignmentParameters* > > alignmentParametersMap;

    if ( readParam || readCovar || applyParam || applyCovar )
    {
      string file = config.getUntrackedParameter< string >( "FileName", "Input.root" );
      int ierr = 0;
      int iter = 1;

      AlignmentIORoot alignmentIO;

      while ( !ierr )
      {
	cout << "[" << *itInitSel << "] read alignment parameters. file / iteration = " << file << " / " << iter << endl;
	vector< AlignmentParameters* > alignmentParameters = alignmentIO.readAlignmentParameters( alignables, file.c_str(), iter, ierr );
	cout << "[" << *itInitSel << "] #param / ierr = " << alignmentParameters.size() << " / " << ierr << endl;

	vector< AlignmentParameters* >::iterator itParam;
	for ( itParam = alignmentParameters.begin(); itParam != alignmentParameters.end(); ++itParam )
	  alignmentParametersMap[(*itParam)->alignable()].push_back( *itParam );

	++iter;
      }
    }

    //
    // Get default values for alignment parameters.
    //
    double sigmaXShift = config.getUntrackedParameter< double >( "SigmaXShifts", 4e-2 );
    double sigmaYShift = config.getUntrackedParameter< double >( "SigmaYShifts", 4e-2 );
    double sigmaZShift = config.getUntrackedParameter< double >( "SigmaZShifts", 4e-2 );
    double sigmaXRot = config.getUntrackedParameter< double >( "SigmaXRotations", 5e-4 );
    double sigmaYRot = config.getUntrackedParameter< double >( "SigmaYRotations", 5e-4 );
    double sigmaZRot = config.getUntrackedParameter< double >( "SigmaZRotations", 5e-4 );

    randSig[0] = sigmaXShift; randSig[1] = sigmaYShift; randSig[2] = sigmaZShift;
    randSig[3] = sigmaXRot; randSig[4] = sigmaYRot; randSig[5] = sigmaZRot;

    startError[0][0] = config.getUntrackedParameter< double >( "XShiftsStartError", 4e-4 );
    startError[1][1] = config.getUntrackedParameter< double >( "YShiftsStartError", 4e-4 );
    startError[2][2] = config.getUntrackedParameter< double >( "ZShiftsStartError", 4e-4 );
    startError[3][3] = config.getUntrackedParameter< double >( "XRotationsStartError", 3e-5 );
    startError[4][4] = config.getUntrackedParameter< double >( "YRotationsStartError", 3e-5 );
    startError[5][5] = config.getUntrackedParameter< double >( "ZRotationsStartError", 3e-5 );

    int iAlign = 0;
    int iApply = 0;

    //
    // Apply initialization to the selected alignables.
    //
    vector< Alignable* >::iterator itAlignable;
    for ( itAlignable = alignables.begin(); itAlignable != alignables.end(); itAlignable++ )
    {
      //
      // Add alignment position error for tracking.
      //
      if ( addPositionError )
      {
	LocalVector localError( sqrt(startError[0][0]), sqrt(startError[1][1]), sqrt(startError[2][2]) );
	GlobalVector globalError = (*itAlignable)->surface().toGlobal( localError );
	AlignmentPositionError ape( globalError.x(), globalError.y(), globalError.z() );
	( *itAlignable )->setAlignmentPositionError( ape );
      }

      //
      // Initialize alignment parameters.
      //
      if ( (*itAlignable)->alignmentParameters() != 0 ) // Read start values for the parameters and errors from file (if found).
      {
	AlignmentParameters* alignmentParameters;
	if ( readParam && readCovar )
	{
	  if ( alignmentParametersMap.find( *itAlignable ) == alignmentParametersMap.end() ) // Not found, use default instead.
	  {
	    alignmentParameters = (*itAlignable)->alignmentParameters()->clone( startParameters, startError );
	    alignmentParameters->setUserVariables( new KalmanAlignmentUserVariables( *itAlignable, alignableId, updateGraph ) );
	  }
	  else
	  {
	    alignmentParameters = alignmentParametersMap[*itAlignable].back();
	    KalmanAlignmentUserVariables* userVariables = new KalmanAlignmentUserVariables( *itAlignable, alignableId, updateGraph );
	    userVariables->update( alignmentParameters );
	    alignmentParameters->setUserVariables( userVariables );
	  }
	}
	else if ( readParam ) // Read start values for the parameters from file (if found) and use default errors.
	{
	  if ( alignmentParametersMap.find( *itAlignable ) == alignmentParametersMap.end() ) // Not found, use default instead.
	  {
	    alignmentParameters = (*itAlignable)->alignmentParameters()->clone( startParameters, startError );
	    alignmentParameters->setUserVariables( new KalmanAlignmentUserVariables( *itAlignable, alignableId, updateGraph ) );
	  }
	  else
	  {
	    AlgebraicVector parameters = alignmentParametersMap[*itAlignable].back()->parameters();
	    alignmentParameters = (*itAlignable)->alignmentParameters()->clone( parameters, startError );
	    KalmanAlignmentUserVariables* userVariables = new KalmanAlignmentUserVariables( *itAlignable, alignableId, updateGraph );
	    userVariables->update( alignmentParameters );
	    alignmentParameters->setUserVariables( userVariables );
	  }
	}
	else // Use default start values for the parameters and errors.
	{
	  alignmentParameters = (*itAlignable)->alignmentParameters()->clone( startParameters, startError );
	  alignmentParameters->setUserVariables( new KalmanAlignmentUserVariables( *itAlignable, alignableId, updateGraph ) );
	}

	(*itAlignable)->setAlignmentParameters( alignmentParameters );

	if ( applyRandomStartValues )
	{
	  AlgebraicVector randomStartParameters = alignmentParameters->parameters();
	  AlgebraicSymMatrix randomStartErrors = alignmentParameters->covariance();

	  for ( int iParam = 0; iParam < randomStartParameters.num_row(); ++iParam )
	  {
	    randomStartParameters[iParam] += sqrt(randSig[iParam])*RandGauss::shoot();
	  }

	  alignmentParameters = (*itAlignable)->alignmentParameters()->clone( randomStartParameters, randomStartErrors );
	  (*itAlignable)->setAlignmentParameters( alignmentParameters );
	}
      }

      //
      // Apply parameters from file to alignables and set corresponding alignment position error.
      //
      if ( ( applyParam || applyCovar ) && alignmentParametersMap.find( *itAlignable ) != alignmentParametersMap.end() )
      {
	++iApply;

	vector< AlignmentParameters* > allAlignmentParameters = alignmentParametersMap[*itAlignable];
	vector< AlignmentParameters* >::iterator itParam;

	// If more than one set of alignment parameters is found, then apply them sequentialy.
	for ( itParam = allAlignmentParameters.begin(); itParam != allAlignmentParameters.end(); ++itParam )
	{
	  RigidBodyAlignmentParameters* alignmentParameters = dynamic_cast<RigidBodyAlignmentParameters*>( *itParam );

	  if ( !alignmentParameters )
	    throw cms::Exception( "BadConfig" ) << "applyParameters: provided alignable does not have rigid body alignment parameters";

	  if ( applyParam )
	  {
	    AlgebraicVector shift = alignmentParameters->translation();
	    const AlignableSurface& alignableSurface = ( *itAlignable )->surface();
	    ( *itAlignable )->move( alignableSurface.toGlobal( align::LocalVector( shift[0], shift[1], shift[2] ) ) );

	    align::EulerAngles angles = alignmentParameters->rotation();
	    if ( angles.normsq() > 1e-10 ) ( *itAlignable )->rotateInLocalFrame( align::toMatrix( angles ) );
	  }

	  if ( applyCovar )
	  {
	    const AlgebraicSymMatrix& aliCov = alignmentParameters->covariance();
	    LocalVector localError( sqrt(aliCov[0][0]), sqrt(aliCov[1][1]), sqrt(aliCov[2][2]) );
	    GlobalVector globalError = (*itAlignable)->surface().toGlobal( localError );
	    AlignmentPositionError ape( globalError.x(), globalError.y(), globalError.z() );
	    ( *itAlignable )->setAlignmentPositionError( ape );
	  }
	}

	KalmanAlignmentUserVariables* userVariables = dynamic_cast< KalmanAlignmentUserVariables* >( ( *itAlignable )->alignmentParameters()->userVariables() );
	if ( userVariables ) { ++iAlign; userVariables->setAlignmentFlag( true ); }
      }
    }

    cout << "[" << *itInitSel << "] Set the alignment flag for " << iAlign << " alignables." << endl;
    cout << "[" << *itInitSel << "] Number of applied parameters: " << iApply << endl; 
    theSelector->clear();
  }
}
