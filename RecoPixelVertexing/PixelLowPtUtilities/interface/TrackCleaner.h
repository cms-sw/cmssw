#ifndef _TrackCleaner_h_
#define _TrackCleaner_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include <utility>
#include <vector>

class TrackerGeometry;

class TrackCleaner : public PixelTrackCleaner
{
  public:
    TrackCleaner (const edm::ParameterSet& ps);
    virtual ~TrackCleaner();

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
   std::vector<const TrackingRecHit*> ttrhs(const SeedingHitSet & h) const;
};

#endif

