#ifndef PixelTrackFitting_PixelTrackCleaner_H
#define PixelTrackFitting_PixelTrackCleaner_H

/**
class PixelTrackCleaner:
Discards reconstructed tracks that reflects one real track.
**/

#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

class TrackerTopology;

class PixelTrackCleaner {

public:

  virtual ~PixelTrackCleaner(){}

  typedef pixeltrackfitting::TracksWithRecHits TracksWithRecHits;
  virtual TracksWithRecHits cleanTracks(const TracksWithRecHits & tracksWithRecHits,
					const TrackerTopology *tTopo) = 0;

private:

};

#endif
