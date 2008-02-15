
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTracklet.h"
#include "DataFormats/TrackReco/interface/Track.h"


KalmanAlignmentTracklet::KalmanAlignmentTracklet( TrajTrackPair& trajTrackPair,
						  const TrajectoryStateOnSurface& external,
						  KalmanAlignmentTrackingSetup* setup ) :
  theTrajTrackPair( trajTrackPair ),
  theExternalPrediction( external ),
  theExternalPredictionFlag( true ),
  theTrackingSetup( setup )
{
  // Reset pointers to NULL.
  trajTrackPair.first = 0;
  trajTrackPair.second = 0;
}


KalmanAlignmentTracklet::KalmanAlignmentTracklet( TrajTrackPair& trajTrackPair,
						  KalmanAlignmentTrackingSetup* setup ) :
  theTrajTrackPair( trajTrackPair ),
  theExternalPredictionFlag( false ),
  theTrackingSetup( setup )
{
  // Reset pointers to NULL.
  trajTrackPair.first = 0;
  trajTrackPair.second = 0;
}



KalmanAlignmentTracklet::~KalmanAlignmentTracklet( void )
{
  delete theTrajTrackPair.first;
  delete theTrajTrackPair.second;
}
