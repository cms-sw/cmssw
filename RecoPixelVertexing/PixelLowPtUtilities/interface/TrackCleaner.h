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
    bool areSame(const TrackingRecHit * a,
                 const TrackingRecHit * b);
    bool isCompatible(const DetId & i1,
                      const DetId & i2);
    bool canBeMerged(std::vector<const TrackingRecHit *> recHitsA,
                     std::vector<const TrackingRecHit *> recHitsB);

    std::vector<const TrackingRecHit*> ttrhs(const SeedingHitSet & h) const;
};

#endif

