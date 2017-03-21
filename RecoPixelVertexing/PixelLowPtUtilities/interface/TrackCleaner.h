#ifndef _TrackCleaner_h_
#define _TrackCleaner_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include <utility>
#include <vector>

class TrackerGeometry;
class TrackerTopology;

class TrackCleaner : public PixelTrackCleaner
{
  public:
    explicit TrackCleaner(const TrackerTopology *tTopo);
    virtual ~TrackCleaner();

    virtual TracksWithRecHits cleanTracks
      (const TracksWithRecHits & tracksWithRecHits) const;

  private:
    bool areSame(const TrackingRecHit * a,
                 const TrackingRecHit * b) const;
    bool isCompatible(const DetId & i1,
                      const DetId & i2) const;
    bool canBeMerged(const std::vector<const TrackingRecHit *>& recHitsA,
                     const std::vector<const TrackingRecHit *>& recHitsB) const;

    std::vector<const TrackingRecHit*> ttrhs(const SeedingHitSet & h) const;

    const TrackerTopology *tTopo_;
};

#endif

