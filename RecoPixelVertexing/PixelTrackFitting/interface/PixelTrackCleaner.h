#ifndef PixelTrackFitting_PixelTrackCleaner_H
#define PixelTrackFitting_PixelTrackCleaner_H

/**
class PixelTrackCleaner:
Discards reconstructed tracks that reflects one real track.
**/

#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include<cassert>

class TrackerTopology;

class PixelTrackCleaner {

public:

  virtual ~PixelTrackCleaner(){}

  // used by HI?
  typedef pixeltrackfitting::TracksWithRecHits TracksWithRecHits;
  virtual TracksWithRecHits cleanTracks(const TracksWithRecHits & tracksWithRecHits, const TrackerTopology *tTopo){
    assert(false); 
    return TracksWithRecHits();
  }


  // fast
  using TracksWithTTRHs = pixeltrackfitting::TracksWithTTRHs;
  virtual void cleanTracks(TracksWithTTRHs & tracksWithRecHits,
					const TrackerTopology *tTopo) const {assert(false);}


  bool fast=false;

};

#endif
