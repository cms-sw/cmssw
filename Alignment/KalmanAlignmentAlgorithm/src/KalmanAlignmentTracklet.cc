
#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentTracklet.h"


KalmanAlignmentTracklet::KalmanAlignmentTracklet( TrajTrackPair& trajTrackPair,
						  const TrajectoryStateOnSurface& external ) :
  theTrajTrackPair( trajTrackPair ),
  theExternalPrediction( external ),
  theExternalPredictionFlag( true )
{
  // Reset pointers to NULL.
  trajTrackPair.first = 0;
  trajTrackPair.second = 0;
}


KalmanAlignmentTracklet::KalmanAlignmentTracklet( TrajTrackPair& trajTrackPair ) :
  theTrajTrackPair( trajTrackPair ),
  theExternalPredictionFlag( false )
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
