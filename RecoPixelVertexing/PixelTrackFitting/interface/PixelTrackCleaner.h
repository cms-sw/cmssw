#ifndef PixelTrackCleaner_H
#define PixelTrackCleaner_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

#include <utility>
#include <vector>


/*
PixelTrackCleaner:

Discards tracks with more than one common recHit.

*/


class PixelTrackCleaner {

public:
     typedef pixeltrackfitting::TracksWithRecHits TracksWithRecHits;
     TracksWithRecHits cleanTracks(const TracksWithRecHits & tracksWithRecHits);

private:

  void cleanTrack();
  bool recHitsAreEqual(const TrackingRecHit *recHit1, const TrackingRecHit *recHit2);

  std::vector<bool> trackOk;
  reco::Track *track1, *track2;
  int iTrack1, iTrack2;

};

#endif
