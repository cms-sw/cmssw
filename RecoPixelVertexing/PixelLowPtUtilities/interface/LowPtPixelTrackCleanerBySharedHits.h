#ifndef _LowPtPixelTrackCleanerBySharedHits_h_
#define _LowPtPixelTrackCleanerBySharedHits_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include <utility>
#include <vector>

class LowPtPixelTrackCleanerBySharedHits : public PixelTrackCleaner
{
  public:
    LowPtPixelTrackCleanerBySharedHits(const edm::ParameterSet& ps);
    virtual ~LowPtPixelTrackCleanerBySharedHits();

    virtual TracksWithRecHits cleanTracks
     (const TracksWithRecHits & tracksWithRecHits);

  private:
    int getLayer(const DetId & id);
    bool hasCommonDetUnit (std::vector<const TrackingRecHit *> recHitsA,
                           std::vector<const TrackingRecHit *> recHitsB,
                           std::vector<DetId> detIds);
    bool hasCommonLayer (std::vector<const TrackingRecHit *> recHitsA,
                         std::vector<const TrackingRecHit *> recHitsB,
                         std::vector<int> detLayers);

};

#endif

