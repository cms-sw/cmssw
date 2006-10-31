#ifndef Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h
#define Alignment_KalmanAlignmentAlgorithm_KalmanAlignmentAlgorithm_h

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentAlgorithmBase.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/TrajectoryFactoryBase.h"

#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentUpdator.h"


/// The main class for the Kalman alignment algorithm. It is the stage on which all the protagonists
/// are playing: the refitter, the trajectory factory and the updator.
/// See E.Widl, R.Fr¨uhwirth, W.Adam, A Kalman Filter for Track-based Alignment, CMS NOTE-2006/022
/// for details.

class AlignableNavigator;
class TrajectoryFitter;


class KalmanAlignmentAlgorithm : public AlignmentAlgorithmBase
{

public:

  typedef TrajectoryFactoryBase::ReferenceTrajectoryCollection ReferenceTrajectoryCollection;

  KalmanAlignmentAlgorithm( const edm::ParameterSet& config );
  virtual ~KalmanAlignmentAlgorithm( void );

  virtual void initialize( const edm::EventSetup& setup, 
			   AlignableTracker* tracker,
			   AlignmentParameterStore* store );

  virtual void terminate( void );

  virtual void run( const edm::EventSetup& setup,
		    const TrajTrackPairCollection& tracks );

  virtual AlgoProductCollection refitTracks( const edm::Event& event,
					     const edm::EventSetup& setup );

private:

  void initializeTrajectoryFitter( const edm::EventSetup& setup );

  edm::ParameterSet theConfiguration;

  TrajectoryFactoryBase* theTrajectoryFactory;
  KalmanAlignmentUpdator* theAlignmentUpdator;

  AlignmentParameterStore* theParameterStore;
  AlignableNavigator* theNavigator;

  TrajectoryFitter* theTrajectoryRefitter;

};

#endif
