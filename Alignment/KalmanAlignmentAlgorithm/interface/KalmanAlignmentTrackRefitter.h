#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentTrackRefitter_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentTrackRefitter_h

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTracklet.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentSetup.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

class TrajectoryFitter;

namespace reco { class BeamSpot; }

/// This class serves the very specific needs of the KalmanAlignmentAlgorithm.
/// Tracks are partially refitted to 'tracklets' using the current estimate on
/// the alignment (see class CurrentAlignmentKFUpdator. These tracklets are
/// either used to compute an exteranal estimate for other tracklets or are
/// handed to the alignment algorithm for further processing. If a tracklet is
/// used as an external prediction or for further processing is defined via
/// the configuration file.
/// NOTE: The trajectory measurements of the tracklets are always ordered along
/// the direction of the momentum!


class KalmanAlignmentTrackRefitter : public TrackProducerBase<reco::Track>
//class KalmanAlignmentTrackRefitter : public TrackProducerBase
{

public:

  typedef std::vector< KalmanAlignmentSetup* > AlignmentSetupCollection;
  typedef KalmanAlignmentSetup::SortingDirection SortingDirection;

  typedef edm::OwnVector< TrackingRecHit > RecHitContainer;

  typedef KalmanAlignmentTracklet::TrajTrackPairCollection TrajTrackPairCollection;
  typedef KalmanAlignmentTracklet::TrackletPtr TrackletPtr;
  typedef std::vector< TrackletPtr > TrackletCollection;

  typedef AlignmentAlgorithmBase::ConstTrajTrackPair ConstTrajTrackPair;
  typedef AlignmentAlgorithmBase::ConstTrajTrackPairCollection ConstTrajTrackPairCollection;

  /// Constructor.
  KalmanAlignmentTrackRefitter( const edm::ParameterSet& config, AlignableNavigator* navigator );

  /// Destructor.
  ~KalmanAlignmentTrackRefitter( void );

  TrackletCollection refitTracks( const edm::EventSetup& eventSetup,
				  const AlignmentSetupCollection& algoSetups,
				  const ConstTrajTrackPairCollection& tracks,
				  const reco::BeamSpot* beamSpot );

  /// Dummy implementation, due to inheritance from TrackProducerBase.
  virtual void produce( edm::Event&, const edm::EventSetup& ) {}

private:

  TrajTrackPairCollection refitSingleTracklet( const TrackingGeometry* geometry,
					       const MagneticField* magneticField,
					       const TrajectoryFitter* fitter,
					       const Propagator* propagator,
					       const TransientTrackingRecHitBuilder* recHitBuilder,
					       const reco::TransientTrack& originalTrack,
					       RecHitContainer& recHits,
					       const reco::BeamSpot* beamSpot,
					       const SortingDirection& sortingDir,
					       bool useExternalEstimate,
					       bool reuseMomentumEstimate,
					       const std::string identifier = std::string("RefitSingle_") );

  void sortRecHits( RecHitContainer& hits,
		    const TransientTrackingRecHitBuilder* builder,
		    const SortingDirection& sortingDir ) const;


  bool rejectTrack( const reco::Track* track ) const;


  void debugTrackData( const std::string identifier,
		       const Trajectory* traj,
		       const reco::Track* track,
		       const reco::BeamSpot* bs );

  TrackProducerAlgorithm<reco::Track> theRefitterAlgo;
  //TrackProducerAlgorithm theRefitterAlgo;
  AlignableNavigator* theNavigator;
  bool theDebugFlag;
};


#endif
