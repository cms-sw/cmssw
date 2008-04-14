
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTrackRefitter.h"

#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentDataCollector.h"

#include "CLHEP/GenericFunctions/CumulativeChiSquare.hh"

#include <iostream>

using namespace std;
using namespace reco;
using namespace Genfun;


KalmanAlignmentTrackRefitter::KalmanAlignmentTrackRefitter( const edm::ParameterSet& config, AlignableNavigator* navigator ) :
  theRefitterAlgo( config ),
  theNavigator( navigator ),
  theDebugFlag( config.getUntrackedParameter<bool>( "debug", true ) )
{
  TrackProducerBase< reco::Track >::setConf( config );
  TrackProducerBase< reco::Track >::setSrc( config.getParameter< edm::InputTag >( "src" ),
					    config.getParameter< edm::InputTag >( "bsSrc" ) );
}


KalmanAlignmentTrackRefitter::~KalmanAlignmentTrackRefitter( void ) {}


KalmanAlignmentTrackRefitter::TrackletCollection
KalmanAlignmentTrackRefitter::refitTracks( const edm::EventSetup& setup,
					   const AlignmentSetupCollection& algoSetups,
					   const ConstTrajTrackPairCollection& tracks )
{
  // Retrieve what we need from the EventSetup
  edm::ESHandle< TrackerGeometry > aGeometry;
  edm::ESHandle< MagneticField > aMagneticField;
  edm::ESHandle< TrajectoryFitter > aTrajectoryFitter;
  edm::ESHandle< Propagator > aPropagator;
  edm::ESHandle< TransientTrackingRecHitBuilder > aRecHitBuilder;

  getFromES( setup, aGeometry, aMagneticField, aTrajectoryFitter, aPropagator, aRecHitBuilder );

  TrackletCollection result;
  TrackCollection fullTracks;

  ConstTrajTrackPairCollection refittedFullTracks;
  ConstTrajTrackPairCollection::const_iterator itTrack;

  for( itTrack = tracks.begin(); itTrack != tracks.end(); ++itTrack )
  {
    TransientTrack fullTrack( *(*itTrack).second, aMagneticField.product() );

    AlignmentSetupCollection::const_iterator itSetup;
    for ( itSetup = algoSetups.begin(); itSetup != algoSetups.end(); ++itSetup )
    {
      RecHitContainer trackingRecHits;
      RecHitContainer externalTrackingRecHits;

      // Extract collection with TrackingRecHits
      Trajectory::ConstRecHitContainer hits = (*itTrack).first->recHits();
      Trajectory::ConstRecHitContainer::iterator itHits;

      for ( itHits = hits.begin(); itHits != hits.end(); ++itHits )
      {
	if ( !(*itHits)->isValid() ) continue;

 	try
	{
	  //if ( !theNavigator->alignableFromDetId( (*itHits)->geographicalId() )->alignmentParameters() ) continue;
	  theNavigator->alignableFromDetId( (*itHits)->geographicalId() );	  
	} catch(...) { continue; }

	if ( (*itSetup)->useForTracking( *itHits ) )
	{
	  trackingRecHits.push_back( (*itHits)->hit()->clone() );
	}
	else if ( (*itSetup)->useForExternalTracking( *itHits ) )
	{
	  externalTrackingRecHits.push_back( (*itHits)->hit()->clone() );
	}
      }

      //edm::LogInfo( "KalmanAlignmentTrackRefitter" ) << "Hits for tracking/external: " << trackingRecHits.size() << "/" << externalTrackingRecHits.size();

      if ( trackingRecHits.empty() ) continue;

      if ( externalTrackingRecHits.empty() )
      {
	if ( ( (*itSetup)->getExternalTrackingSubDetIds().size() == 0 ) && // O.K., no external hits expected,
	     ( trackingRecHits.size() >= (*itSetup)->minTrackingHits() ) )
	{
	  TrajTrackPairCollection refitted = refitSingleTracklet( aGeometry.product(), aMagneticField.product(),
								  (*itSetup)->fitter(), (*itSetup)->propagator(), 
								  aRecHitBuilder.product(), fullTrack,
								  trackingRecHits, (*itSetup)->sortingDirection(),
								  false, true );

	  if ( refitted.empty() ) continue; // The refitting did not work ... Try next!

	  if ( theDebugFlag )
	  {
	    debugTrackData( (*itSetup)->id(), refitted.front().first, refitted.front().second );
	    debugTrackData( "OrigFullTrack", (*itTrack).first, (*itTrack).second );
	  }


	  TrackletPtr trackletPtr( new KalmanAlignmentTracklet( refitted.front(), *itSetup ) );
	  result.push_back( trackletPtr );
	}
	else { continue; } // Expected external hits but found none or not enough hits.
      }
      else if ( ( trackingRecHits.size() >= (*itSetup)->minTrackingHits() ) &&
		( externalTrackingRecHits.size() >= (*itSetup)->minExternalHits() ) ) 
      {
	// Create an instance of KalmanAlignmentTracklet with an external prediction.

	TrajTrackPairCollection external = refitSingleTracklet( aGeometry.product(), aMagneticField.product(),
								(*itSetup)->externalFitter(), (*itSetup)->externalPropagator(),
								aRecHitBuilder.product(), fullTrack,
								externalTrackingRecHits, (*itSetup)->externalSortingDirection(),
								false, true );
	if ( external.empty() ) { continue; }

	TransientTrack externalTrack( *external.front().second, aMagneticField.product() );

	TrajTrackPairCollection refitted = refitSingleTracklet( aGeometry.product(), aMagneticField.product(),
								(*itSetup)->fitter(), (*itSetup)->propagator(),
								aRecHitBuilder.product(), externalTrack,
								trackingRecHits, (*itSetup)->sortingDirection(),
								false, true );
	if ( refitted.empty() ) continue;

 	//const Surface& surface = refitted.front().first->firstMeasurement().updatedState().surface();
	const Surface& surface = refitted.front().first->lastMeasurement().updatedState().surface();
 	TrajectoryStateOnSurface externalTsos = externalTrack.impactPointState();
	AnalyticalPropagator externalPredictionPropagator( aMagneticField.product(), anyDirection );
 	TrajectoryStateOnSurface externalPrediction = externalPredictionPropagator.propagate( externalTsos, surface );
	if ( !externalPrediction.isValid() ) continue;

	if ( theDebugFlag )
	{
	  debugTrackData( string("External") + (*itSetup)->id(), external.front().first, external.front().second );
	  debugTrackData( (*itSetup)->id(), refitted.front().first, refitted.front().second );
	  debugTrackData( "OrigFullTrack", (*itTrack).first, (*itTrack).second );
	}

 	TrackletPtr trackletPtr( new KalmanAlignmentTracklet( refitted.front(), externalPrediction, *itSetup ) );
	result.push_back( trackletPtr );

	delete external.front().first;
	delete external.front().second;
      }
    }
  }

  return result;
}


KalmanAlignmentTrackRefitter::TrajTrackPairCollection
KalmanAlignmentTrackRefitter::refitSingleTracklet( const TrackingGeometry* geometry,
						   const MagneticField* magneticField,
						   const TrajectoryFitter* fitter,
						   const Propagator* propagator,
						   const TransientTrackingRecHitBuilder* recHitBuilder,
						   const TransientTrack& fullTrack,
						   RecHitContainer& recHits,
						   const SortingDirection& sortingDir,
						   bool useExternalEstimate,
						   bool reuseMomentumEstimate )
{

  TrajTrackPairCollection result;

  if ( recHits.size() < 2 ) return result;

  sortRecHits( recHits, recHitBuilder, sortingDir );

  TransientTrackingRecHit::RecHitPointer firstHit = recHitBuilder->build( &recHits.front() );

  AnalyticalPropagator firstStatePropagator( magneticField, anyDirection );
  TrajectoryStateOnSurface firstState = firstStatePropagator.propagate( fullTrack.impactPointState(), firstHit->det()->surface() );

  if ( !firstState.isValid() ) return result;

  const double startErrorValue = 100;
  const unsigned int nTrajParam = 5;

  LocalTrajectoryError startError;

  if ( useExternalEstimate ) {
    startError = firstState.localError();
  } else {
    if ( reuseMomentumEstimate )
    {
      AlgebraicSymMatrix firstStateError( asHepMatrix( firstState.localError().matrix() ) );
      AlgebraicSymMatrix startErrorMatrix( nTrajParam, 0 );
      startErrorMatrix[0][0] = firstStateError[0][0];
      startErrorMatrix[1][1] = startErrorValue;//firstStateError[1][1];
      startErrorMatrix[2][2] = startErrorValue;//firstStateError[2][2];
      startErrorMatrix[3][3] = startErrorValue;
      startErrorMatrix[4][4] = startErrorValue;
      startError = LocalTrajectoryError( startErrorMatrix );
    } else {
      AlgebraicSymMatrix startErrorMatrix( startErrorValue*AlgebraicSymMatrix( nTrajParam, 1 ) );
      startError = LocalTrajectoryError( startErrorMatrix );
    }

  }

//   // MOMENTUM ESTIMATE FOR COSMICS. P = 1.5 GeV
//   LocalTrajectoryParameters firstStateParameters = firstState.localParameters();
//   AlgebraicVector firstStateParamVec = asHepVector( firstStateParameters.mixedFormatVector() );
//   firstStateParamVec[0] = 1./1.5;
//   LocalTrajectoryParameters cosmicsStateParameters( firstStateParamVec, firstStateParameters.pzSign(), true );
//   TrajectoryStateOnSurface tsos( cosmicsStateParameters, startError, firstState.surface(), magneticField );

  TrajectoryStateOnSurface tsos( firstState.localParameters(), startError, firstState.surface(), magneticField );

  TrajectoryStateTransform stateTransform;
  PTrajectoryStateOnDet state = *stateTransform.persistentState( tsos, firstHit->det()->geographicalId().rawId() );

  // Generate a trajectory seed.
  TrajectorySeed seed( state, recHits, propagator->propagationDirection() );

  // Generate track candidate.
  TrackCandidate candidate( recHits, seed, state );

  TrackCandidateCollection candidateCollection;
  candidateCollection.push_back( candidate );

  // Only dummy implementation of beam spot constraint.
  reco::BeamSpot dummyBeamSpot;
  dummyBeamSpot.dummy();

  AlgoProductCollection algoResult;

  theRefitterAlgo.runWithCandidate( geometry, magneticField, candidateCollection,
				    fitter, propagator, recHitBuilder, dummyBeamSpot,
				    algoResult );

  for ( AlgoProductCollection::iterator it = algoResult.begin(); it != algoResult.end(); ++it )
    result.push_back( make_pair( (*it).first, (*it).second.first ) );

  return result;
}


void KalmanAlignmentTrackRefitter::sortRecHits( RecHitContainer& hits,
						const TransientTrackingRecHitBuilder* builder,
						const SortingDirection& sortingDir ) const
{
  // Don't start sorting if there is only 1 or even 0 elements.
  if ( hits.size() < 2 ) return;

  TransientTrackingRecHit::RecHitPointer firstHit = builder->build( &hits.front() );
  double firstRadius = firstHit->det()->surface().toGlobal( firstHit->localPosition() ).mag();
  double firstY = firstHit->det()->surface().toGlobal( firstHit->localPosition() ).y();

  TransientTrackingRecHit::RecHitPointer lastHit = builder->build( &hits.back() );
  double lastRadius = lastHit->det()->surface().toGlobal( lastHit->localPosition() ).mag();
  double lastY = lastHit->det()->surface().toGlobal( lastHit->localPosition() ).y();

  bool insideOut = firstRadius < lastRadius;
  bool upsideDown = lastY < firstY;

  if ( ( insideOut && ( sortingDir == KalmanAlignmentSetup::sortInsideOut ) ) ||
       ( !insideOut && ( sortingDir == KalmanAlignmentSetup::sortOutsideIn ) ) ||
       ( upsideDown && ( sortingDir == KalmanAlignmentSetup::sortUpsideDown ) ) ||
       ( !upsideDown && ( sortingDir == KalmanAlignmentSetup::sortDownsideUp ) ) ) return;

  // Fill temporary container with reversed hits.
  RecHitContainer tmp;
  RecHitContainer::iterator itHit = hits.end();
  do { --itHit; tmp.push_back( ( *itHit ).clone() ); } while ( itHit != hits.begin() );

  // Swap the content of the temporary and the input container.
  hits.swap( tmp );

  return;
}


void KalmanAlignmentTrackRefitter::debugTrackData( const string identifier, const Trajectory* traj, const Track* track )
{
  GENFUNCTION cumulativeChi2 = CumulativeChiSquare( static_cast<unsigned int>( track->ndof() ) );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_CumChi2"), 1. - cumulativeChi2( track->chi2() ) );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_NHits"), traj->foundHits() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_Pt"), 1e-2*track->pt() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_Eta"), track->eta() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_Phi"), track->phi() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_NormChi2"), track->normalizedChi2() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_DZ"), track->dz() );
}
