
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTrackRefitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TRecHit1DMomConstraint.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

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
  // --- GPetrucc: I can't understand where anything is read from the event, and who's the consumer.
  //               If there is one anywhere, it should do the consumes<T> calls and pass that to the setSrc
  //TrackProducerBase< reco::Track >::setSrc( consumes<TrackCandidateCollection>(iConfig.getParameter<edm::InputTag>( "src" )),
  //                                          consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>( "beamSpot" )));
}


KalmanAlignmentTrackRefitter::~KalmanAlignmentTrackRefitter( void ) {}


KalmanAlignmentTrackRefitter::TrackletCollection
KalmanAlignmentTrackRefitter::refitTracks( const edm::EventSetup& setup,
					   const AlignmentSetupCollection& algoSetups,
					   const ConstTrajTrackPairCollection& tracks,
					   const reco::BeamSpot* beamSpot )
{
  // Retrieve what we need from the EventSetup
  edm::ESHandle< TrackerGeometry > aGeometry;
  edm::ESHandle< MagneticField > aMagneticField;
  edm::ESHandle< TrajectoryFitter > aTrajectoryFitter;
  edm::ESHandle< Propagator > aPropagator;
  edm::ESHandle<MeasurementTracker> theMeasTk;
  edm::ESHandle< TransientTrackingRecHitBuilder > aRecHitBuilder;

  getFromES( setup,
             aGeometry,
             aMagneticField,
             aTrajectoryFitter,
             aPropagator,
             theMeasTk,
             aRecHitBuilder );

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

      ////RecHitContainer pixelRecHits;
      RecHitContainer zPlusRecHits;
      RecHitContainer zMinusRecHits;

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

	  ( (*itHits)->det()->position().z() > 0. ) ?
	    zPlusRecHits.push_back( (*itHits)->hit()->clone() ) :
	    zMinusRecHits.push_back( (*itHits)->hit()->clone() );

	  ////const int subdetId( (*itHits)->det()->geographicalId().subdetId() );
	  ////if ( subdetId == 1 ) pixelRecHits.push_back( (*itHits)->hit()->clone() );
	}
	else if ( (*itSetup)->useForExternalTracking( *itHits ) )
	{
	  externalTrackingRecHits.push_back( (*itHits)->hit()->clone() );
	}
      }

      //edm::LogInfo( "KalmanAlignmentTrackRefitter" ) << "Hits for tracking/external: " << trackingRecHits.size() << "/" << externalTrackingRecHits.size();

      //if ( !zPlusRecHits.size() || !zMinusRecHits.size() ) continue;
      ////if ( pixelRecHits.size() < 3 || !zPlusRecHits.size() || !zMinusRecHits.size() ) continue;

      if ( trackingRecHits.empty() ) continue;

      if ( externalTrackingRecHits.empty() )
      {
	if ( ( (*itSetup)->getExternalTrackingSubDetIds().size() == 0 ) && // O.K., no external hits expected,
	     ( trackingRecHits.size() >= (*itSetup)->minTrackingHits() ) )
	{
	  TrajTrackPairCollection refitted = refitSingleTracklet( aGeometry.product(), aMagneticField.product(),
								  (*itSetup)->fitter(), (*itSetup)->propagator(),
								  aRecHitBuilder.product(), fullTrack,
								  trackingRecHits, beamSpot,
								  (*itSetup)->sortingDirection(), false, true );

	  // The refitting did not work ... Try next!
	  if ( refitted.empty() ) continue;
	  if ( rejectTrack( refitted.front().second ) ) continue;

	  if ( theDebugFlag )
	  {
	    debugTrackData( (*itSetup)->id(), refitted.front().first, refitted.front().second, beamSpot );
	    debugTrackData( "OrigFullTrack", (*itTrack).first, (*itTrack).second, beamSpot );
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
								externalTrackingRecHits, beamSpot,
								(*itSetup)->externalSortingDirection(),
								false, true );
	//if ( external.empty() || rejectTrack( external.front().second ) ) { continue; }
	if ( external.empty() ) { continue; }

	TransientTrack externalTrack( *external.front().second, aMagneticField.product() );

	TrajTrackPairCollection refitted = refitSingleTracklet( aGeometry.product(), aMagneticField.product(),
								(*itSetup)->fitter(), (*itSetup)->propagator(),
								aRecHitBuilder.product(), externalTrack,
								trackingRecHits, beamSpot,
								(*itSetup)->sortingDirection(),
								false, true, (*itSetup)->id() );

	if ( refitted.empty() ) { continue; }
	if ( rejectTrack( refitted.front().second ) ) continue;

 	//const Surface& surface = refitted.front().first->firstMeasurement().updatedState().surface();
	const Surface& surface = refitted.front().first->lastMeasurement().updatedState().surface();
 	TrajectoryStateOnSurface externalTsos = externalTrack.impactPointState();
	AnalyticalPropagator externalPredictionPropagator( aMagneticField.product(), anyDirection );
 	TrajectoryStateOnSurface externalPrediction = externalPredictionPropagator.propagate( externalTsos, surface );
	if ( !externalPrediction.isValid() ) continue;

	if ( theDebugFlag )
	{
	  debugTrackData( string("External") + (*itSetup)->id(), external.front().first, external.front().second, beamSpot );
	  debugTrackData( (*itSetup)->id(), refitted.front().first, refitted.front().second, beamSpot );
	  debugTrackData( "OrigFullTrack", (*itTrack).first, (*itTrack).second, beamSpot );
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
						   const reco::BeamSpot* beamSpot,
						   const SortingDirection& sortingDir,
						   bool useExternalEstimate,
						   bool reuseMomentumEstimate,
						   const string identifier )
{

  TrajTrackPairCollection result;

  if ( recHits.size() < 2 ) return result;

  sortRecHits( recHits, recHitBuilder, sortingDir );

  TransientTrackingRecHit::RecHitContainer hits;
  RecHitContainer::iterator itRecHit;
  for ( itRecHit = recHits.begin(); itRecHit != recHits.end(); ++itRecHit )
    hits.push_back( recHitBuilder->build( &(*itRecHit) ) );

  TransientTrackingRecHit::ConstRecHitPointer firstHit = hits.front();

  AnalyticalPropagator firstStatePropagator( magneticField, anyDirection );
  TrajectoryStateOnSurface firstState = firstStatePropagator.propagate( fullTrack.impactPointState(), firstHit->det()->surface() );

  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_IPPt"),
					       1e-2*fullTrack.impactPointState().globalParameters().momentum().perp() );

  if ( !firstState.isValid() ) return result;

//   LocalTrajectoryError startError;

//   const double startErrorValue = 100;
//   const unsigned int nTrajParam = 5;

//   if ( useExternalEstimate ) {
//     startError = firstState.localError();
//   } else {
//     if ( reuseMomentumEstimate )
//     {
//       AlgebraicSymMatrix firstStateError( asHepMatrix( firstState.localError().matrix() ) );
//       AlgebraicSymMatrix startErrorMatrix( nTrajParam, 0 );
//       startErrorMatrix[0][0] = 1e-10;
//       //startErrorMatrix[0][0] = firstStateError[0][0];
//       startErrorMatrix[1][1] = startErrorValue;//firstStateError[1][1];
//       startErrorMatrix[2][2] = startErrorValue;//firstStateError[2][2];
//       startErrorMatrix[3][3] = startErrorValue;
//       startErrorMatrix[4][4] = startErrorValue;
//       startError = LocalTrajectoryError( startErrorMatrix );
//     } else {
//       AlgebraicSymMatrix startErrorMatrix( startErrorValue*AlgebraicSymMatrix( nTrajParam, 1 ) );
//       startError = LocalTrajectoryError( startErrorMatrix );
//     }

//   }

//   // MOMENTUM ESTIMATE FOR COSMICS. P = 1.5 GeV
//   LocalTrajectoryParameters firstStateParameters = firstState.localParameters();
//   AlgebraicVector firstStateParamVec = asHepVector( firstStateParameters.mixedFormatVector() );
//   firstStateParamVec[0] = 1./1.5;
//   LocalTrajectoryParameters cosmicsStateParameters( firstStateParamVec, firstStateParameters.pzSign(), true );
//   TrajectoryStateOnSurface tsos( cosmicsStateParameters, startError, firstState.surface(), magneticField );

  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_FSPt"),
					       1e-2*firstState.globalParameters().momentum().perp() );

  firstState.rescaleError( 100 );
  TrajectoryStateOnSurface tsos( firstState.localParameters(), firstState.localError(),
				 firstState.surface(), magneticField );

  // Generate a trajectory seed.
  TrajectorySeed seed( PTrajectoryStateOnDet(), recHits, propagator->propagationDirection() );

  // Generate track candidate.

  PTrajectoryStateOnDet state = trajectoryStateTransform::persistentState( tsos, firstHit->det()->geographicalId().rawId() );
  TrackCandidate candidate( recHits, seed, state );

  AlgoProductCollection algoResult;

  int charge = static_cast<int>( tsos.charge() );
  double momentum = firstState.localParameters().momentum().mag();
  TransientTrackingRecHit::RecHitPointer testhit =
    TRecHit1DMomConstraint::build( charge, momentum, 1e-10, &tsos.surface() );

  //no insert in OwnVector...
  TransientTrackingRecHit::RecHitContainer tmpHits;
  tmpHits.push_back(testhit);
  for (TransientTrackingRecHit::RecHitContainer::const_iterator i=hits.begin(); i!=hits.end(); i++){
    tmpHits.push_back(*i);
  }
  hits.swap(tmpHits);

  theRefitterAlgo.buildTrack( fitter, propagator, algoResult, hits, tsos, seed, 0, *beamSpot, candidate.seedRef());

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


bool KalmanAlignmentTrackRefitter::rejectTrack( const Track* track ) const
{
  double trackChi2 = track->chi2();
  unsigned int ndof = static_cast<unsigned int>( track->ndof() );
  if ( trackChi2 <= 0. || ndof <= 0 ) return false;

  //FIXME: should be configurable (via KalmanAlignmentSetup)
  double minChi2Prob = 0;//1e-6;
  double maxChi2Prob = 1.0;

  GENFUNCTION cumulativeChi2 = Genfun::CumulativeChiSquare( ndof );
  double chi2Prob = 1. - cumulativeChi2( trackChi2 );
  return ( chi2Prob < minChi2Prob ) || ( chi2Prob > maxChi2Prob );
}


void KalmanAlignmentTrackRefitter::debugTrackData( const string identifier,
						   const Trajectory* traj,
						   const Track* track,
						   const reco::BeamSpot* bs )
{
  unsigned int ndof = static_cast<unsigned int>( track->ndof() );
  double trackChi2 = track->chi2();
  if ( ( trackChi2 > 0. ) && ( ndof > 0 ) )
  {
    GENFUNCTION cumulativeChi2 = Genfun::CumulativeChiSquare( ndof );
    KalmanAlignmentDataCollector::fillHistogram( identifier + string("_CumChi2"), 1. - cumulativeChi2( trackChi2 ) );
  } else if ( ndof == 0 ) {
    KalmanAlignmentDataCollector::fillHistogram( identifier + string("_CumChi2"), -1. );
  } else {
    KalmanAlignmentDataCollector::fillHistogram( identifier + string("_CumChi2"), -2. );
  }

  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_NHits"), traj->foundHits() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_Pt"), 1e-2*track->pt() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_Eta"), track->eta() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_Phi"), track->phi() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_NormChi2"), track->normalizedChi2() );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_DZ"), track->dz() );

  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_DXY_BS"), fabs( track->dxy( bs->position() ) ) );
  KalmanAlignmentDataCollector::fillHistogram( identifier + string("_DXY"), fabs( track->dxy() ) );
  //KalmanAlignmentDataCollector::fillHistogram( identifier + string("_D0"), fabs( track->d0() ) );
}
