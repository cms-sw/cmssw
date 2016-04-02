#ifndef PixelTrackFitting_PixelTrackCleanerBySharedHits_H
#define PixelTrackFitting_PixelTrackCleanerBySharedHits_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include <utility>
#include <vector>


class TrackerTopology;

class PixelTrackCleanerBySharedHits : public PixelTrackCleaner {

public:
  PixelTrackCleanerBySharedHits( const edm::ParameterSet& cfg);

  virtual ~PixelTrackCleanerBySharedHits();

  typedef pixeltrackfitting::TracksWithRecHits TracksWithRecHits;
  virtual TracksWithRecHits cleanTracks(const TracksWithRecHits & tracksWithRecHits, const TrackerTopology *tTopo);

private:
  const bool useQuadrupletAlgo_;
};

#endif
