#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"

// include for refitting
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerAlgorithm.h"

/// The main class for the Kalman alignment algorithm. It is the stage on which all the protagonists
/// are playing: the refitter, the trajectory factory and the updator.
/// See E.Widl, R.Fr¨uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS NOTE-2006/022
/// for details.

class AlignableNavigator;
class TrajectoryFitter;


class KalmanAlignmentAlgorithm : public AlignmentAlgorithmBase, public TrackProducerBase<reco::Track>
{

public:

  typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection ReferenceTrajectoryCollection;

  KalmanAlignmentAlgorithm( const edm::ParameterSet& config );
  virtual ~KalmanAlignmentAlgorithm( void );

  /// Dummy implementation.
  /// Needed for inheritance of TrackProducerBase, but we don't produce anything.
  virtual void produce( edm::Event&, const edm::EventSetup& ) {}

  virtual void initialize( const edm::EventSetup& setup, 
			   AlignableTracker* tracker,
                           AlignableMuon* muon,
			   AlignmentParameterStore* store );

  virtual void terminate( void );

  virtual void run( const edm::EventSetup& setup,
		    const ConstTrajTrackPairCollection& tracks );

private:

  void initializeTrajectoryFitter( const edm::EventSetup& setup );

  void initializeAlignmentParameters( const edm::EventSetup& setup );

  ConstTrajTrackPairCollection refitTracks( const edm::EventSetup& setup,
					    const ConstTrajTrackPairCollection& tracks );

  edm::ParameterSet theConfiguration;

  TrajectoryFactoryBase* theTrajectoryFactory;
  KalmanAlignmentUpdator* theAlignmentUpdator;
  KalmanAlignmentMetricsUpdator* theMetricsUpdator;

  AlignmentParameterStore* theParameterStore;
  AlignableNavigator* theNavigator;

  TrackProducerAlgorithm<reco::Track> theRefitterAlgo;
  TrajectoryFitter* theTrajectoryRefitter;
  bool theRefitterDebugFlag;
};

#endif
