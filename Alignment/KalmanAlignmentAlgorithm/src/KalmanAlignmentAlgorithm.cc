
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentAlgorithm.h"

// includes for alignment
#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryPlugin.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdatorPlugin.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentInitialization.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUserVariables.h"

// includes for retracking
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectorySmoother.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/CurrentAlignmentKFUpdator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/DataCollector.h"

using namespace edm;
using namespace alignmentservices;


KalmanAlignmentAlgorithm::KalmanAlignmentAlgorithm( const edm::ParameterSet& config ) :
  AlignmentAlgorithmBase( config ),
  theConfiguration( config )
{}


KalmanAlignmentAlgorithm::~KalmanAlignmentAlgorithm( void ) {}


void KalmanAlignmentAlgorithm::initialize( const edm::EventSetup& setup, 
					   AlignableTracker* tracker, 
					   AlignmentParameterStore* store )
{
  theParameterStore = store;
  theNavigator = new AlignableNavigator( tracker->components() );

  initializeTrajectoryFitter( setup );

  const ParameterSet initConfig = theConfiguration.getParameter< ParameterSet >( "Initialization" );
  KalmanAlignmentInitialization init( initConfig );
  init.initializeAlignmentParameters( theParameterStore );

  std::string updator = theConfiguration.getParameter< std::string >( "AlignmentUpdator" );
  const ParameterSet updatorConfig = theConfiguration.getParameter< ParameterSet >( updator );
  theAlignmentUpdator = KalmanAlignmentUpdatorPlugin::getUpdator( updator, updatorConfig );

  std::string factory = theConfiguration.getParameter< std::string >( "TrajectoryFactory" );
  const ParameterSet factoryConfig = theConfiguration.getParameter< ParameterSet >( factory );
  theTrajectoryFactory = TrajectoryFactoryPlugin::getFactory( factory, factoryConfig );
}


void KalmanAlignmentAlgorithm::terminate( void )
{
  std::cout << "[KalmanAlignmentAlgorithm::terminate] start ..." << std::endl;

  if ( DataCollector::isAvailable() ) DataCollector::write( "test.root" );
  
  delete theNavigator;
  delete theTrajectoryRefitter;
  delete theTrajectoryFactory;
  delete theAlignmentUpdator;

  std::cout << "[KalmanAlignmentAlgorithm::terminate] ... done." << std::endl;
}


void KalmanAlignmentAlgorithm::run( const edm::EventSetup & setup,
				    const TrajTrackPairCollection & tracks )
{
  static int iEvent = 1;
  std::cout << "[KalmanAlignmentAlgorithm::run] Event Nr. " << iEvent << std::endl;
  iEvent++;

  ReferenceTrajectoryCollection trajectories = theTrajectoryFactory->trajectories( setup, tracks );
  ReferenceTrajectoryCollection::iterator itTrajectories = trajectories.begin();

  while( itTrajectories != trajectories.end() )
  {
    theAlignmentUpdator->process( *itTrajectories, theParameterStore, theNavigator );
    ++itTrajectories;
  }
}


AlgoProductCollection KalmanAlignmentAlgorithm::refitTracks( const edm::Event& event,
							     const edm::EventSetup& setup )
{
  // Retrieve what we need from the EventSetup
  edm::ESHandle<TrackerGeometry>  aGeometry;
  edm::ESHandle<MagneticField>    aMagneticField;
  edm::ESHandle<TrajectoryFitter> aTrajectoryFitter;
  edm::ESHandle<Propagator>       aPropagator;
  edm::ESHandle<TransientTrackingRecHitBuilder> aRecHitBuilder;

  getFromES( setup, aGeometry, aMagneticField, aTrajectoryFitter, aPropagator, aRecHitBuilder );

  // Retrieve track collection from the event
  edm::Handle< reco::TrackCollection > aTrackCollection;
  event.getByLabel( theSrc, aTrackCollection );

  // Dump original tracks
  vector< float > origPt;
  vector< float > origEta;
  vector< float > origPhi;
  vector< float > origChi2;
  if ( 1 )
  {
    for( reco::TrackCollection::const_iterator itrack = aTrackCollection->begin(); itrack != aTrackCollection->end(); ++itrack )
    {
      reco::Track track=*itrack;
      DataCollector::fillHistogram( "OrigTrack_Pt", track.pt() );
      DataCollector::fillHistogram( "OrigTrack_Eta", track.eta() );
      DataCollector::fillHistogram( "OrigTrack_Phi", track.phi() );
      DataCollector::fillHistogram( "OrigTrack_Chi2", track.normalizedChi2() );

      origPt.push_back( track.pt() );
      origEta.push_back( track.eta() );
      origPhi.push_back( track.phi() );
      origChi2.push_back( track.normalizedChi2() );
    }
  }

  AlgoProductCollection algoResults;
  theRefitterAlgo.runWithTrack( aGeometry.product(), aMagneticField.product(), *aTrackCollection,
				theTrajectoryRefitter, aPropagator.product(),
				aRecHitBuilder.product(), algoResults );

  // Dump refitted tracks
  if ( 1 ) 
  {
    int iTrack = 0;
    for( AlgoProductCollection::const_iterator it = algoResults.begin(); it!=algoResults.end(); it++ )
    {
      //Trajectory* traj = (*it).first;
      reco::Track* track = (*it).second;
      DataCollector::fillHistogram( "RefitTrack_Pt", track->pt() );
      DataCollector::fillHistogram( "RefitTrack_Eta", track->eta() );
      DataCollector::fillHistogram( "RefitTrack_Phi", track->phi() );
      DataCollector::fillHistogram( "RefitTrack_Chi2", track->normalizedChi2() );

      DataCollector::fillHistogram( "Track_RelativeDeltaPt", ( track->pt() - origPt[iTrack] )/origPt[iTrack] );
      DataCollector::fillHistogram( "Track_DeltaEta", track->eta() - origEta[iTrack] );
      DataCollector::fillHistogram( "Track_DeltaPhi", track->phi() - origPhi[iTrack] );
      DataCollector::fillHistogram( "Track_DeltaChi2", track->normalizedChi2() - origChi2[iTrack] );

      ++iTrack;
    }
  }

  return algoResults;

}


void KalmanAlignmentAlgorithm::initializeTrajectoryFitter( const edm::EventSetup& setup )
{
  edm::ESHandle< TrajectoryFitter > defaultTrajectoryFitter;
  std::string fitterName = theConfiguration.getParameter< std::string >( "Fitter" );
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
