#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/ReferenceTrajectories/interface/TrajectoryFactoryBase.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsUpdator.h"
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTrackRefitter.h"

#include <set>

/// The main class for the Kalman alignment algorithm. It is the stage on which all the protagonists
/// are playing: the refitter, the trajectory factory and the updator.
/// See E.Widl, R.Fr¨uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS NOTE-2006/022
/// for details.

class AlignableNavigator;
class AlignmentParameterSelector;
class TrajectoryFitter;

class KalmanAlignmentAlgorithm : public AlignmentAlgorithmBase
{

public:

  typedef TrajectoryFactoryBase::ReferenceTrajectoryPtr ReferenceTrajectoryPtr;
  typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection ReferenceTrajectoryCollection;
  typedef TrajectoryFactoryBase::ExternalPredictionCollection ExternalPredictionCollection;

  typedef KalmanAlignmentTracklet::TrackletPtr TrackletPtr;
  typedef std::vector< TrackletPtr > TrackletCollection;


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

  inline bool operator()( const Alignable* a1, const Alignable* a2 ) const { return ( a1->id() < a2->id() ); }

private:

  void initializeAlignmentParameters( const edm::EventSetup& setup );

  edm::ParameterSet theConfiguration;

  AlignmentParameterStore* theParameterStore;
  AlignableNavigator* theNavigator;
  AlignmentParameterSelector* theSelector;

  KalmanAlignmentTrackRefitter* theRefitter;

  TrajectoryFactoryBase* theTrajectoryFactory;

  KalmanAlignmentUpdator* theAlignmentUpdator;
  KalmanAlignmentMetricsUpdator* theMetricsUpdator;
};

#endif
