
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTracklet.h"
#include "DataFormats/TrackReco/interface/Track.h"


KalmanAlignmentTracklet::KalmanAlignmentTracklet( TrajTrackPair& trajTrackPair,
						  const TrajectoryStateOnSurface& external,
						  KalmanAlignmentSetup* setup ) :
  theTrajTrackPair( trajTrackPair ),
  theExternalPrediction( external ),
  theExternalPredictionFlag( true ),
  theAlignmentSetup( setup )
{
  // Reset pointers to NULL.
  trajTrackPair.first = 0;
  trajTrackPair.second = 0;
}


KalmanAlignmentTracklet::KalmanAlignmentTracklet( TrajTrackPair& trajTrackPair,
						  KalmanAlignmentSetup* setup ) :
  theTrajTrackPair( trajTrackPair ),
  theExternalPredictionFlag( false ),
  theAlignmentSetup( setup )
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
