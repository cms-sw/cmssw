#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentTrackRefitter_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentTrackRefitter_h

#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

#include "Alignment/CommonAlignment/interface/AlignableNavigator.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTracklet.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTrackingSetup.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

class TrajectoryFitter;

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
{

public:

  typedef KalmanAlignmentTrackingSetup TrackingSetup;
  typedef std::vector< TrackingSetup* > TrackingSetupCollection;

  typedef edm::OwnVector< TrackingRecHit > RecHitContainer;

  typedef KalmanAlignmentTracklet::TrajTrackPairCollection TrajTrackPairCollection;
  typedef KalmanAlignmentTracklet::TrackletPtr TrackletPtr;
  typedef std::vector< TrackletPtr > TrackletCollection;

  typedef AlignmentAlgorithmBase::ConstTrajTrackPair ConstTrajTrackPair;
  typedef AlignmentAlgorithmBase::ConstTrajTrackPairCollection ConstTrajTrackPairCollection;

  /// Constructor.
  KalmanAlignmentTrackRefitter( const edm::ParameterSet& config );

  /// Destructor.
  ~KalmanAlignmentTrackRefitter( void );

  void initialize( const edm::EventSetup& setup, AlignableNavigator* navigator );

  TrackletCollection refitTracks( const edm::EventSetup& setup,
				  const ConstTrajTrackPairCollection& tracks );

  const TrackingSetupCollection& getTrackingSetups( void ) const { return theTrackingSetup; }

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
					       bool useExternalEstimate,
					       bool reuseMomentumEstimate,
					       bool sortInsideOut );

  void sortRecHits( RecHitContainer& hits,
		    const TransientTrackingRecHitBuilder* builder,
		    bool sortInsideOut ) const;

  void debugTrackData( const std::string identifier, const Trajectory* traj, const reco::Track* track );

  inline const PropagationDirection getDirection( const std::string& dir )
    { return ( dir == "alongMomentum" ) ? alongMomentum : oppositeToMomentum; }

  inline const PropagationDirection oppositeDirection( const PropagationDirection dir )
    { return ( dir == alongMomentum ) ? oppositeToMomentum : alongMomentum; }

  const edm::ParameterSet theConfiguration;

  TrackingSetupCollection theTrackingSetup;

  TrackProducerAlgorithm<reco::Track> theRefitterAlgo;

  AlignableNavigator* theNavigator;

  bool theDebugFlag;
};


#endif
