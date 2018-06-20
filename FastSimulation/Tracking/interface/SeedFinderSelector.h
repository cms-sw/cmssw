#ifndef SEEDFINDERSELECTOR_H
#define SEEDFINDERSELECTOR_H

#include <vector>
#include <memory>
#include <string>

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingLayerSetsBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class TrackingRegion;
class FastTrackerRecHit;
class MultiHitGeneratorFromPairAndLayers;
class HitTripletGeneratorFromPairAndLayers;
class MeasurementTracker;
class CAHitTripletGenerator;
class CAHitQuadrupletGenerator;

namespace edm
{
    class Event;
    class EventSetup;
    class ParameterSet;
    class ConsumesCollector;
}

class SeedFinderSelector
{
public:

    SeedFinderSelector(const edm::ParameterSet & cfg,edm::ConsumesCollector && consumesCollector);
    
    ~SeedFinderSelector();

    void initEvent(const edm::Event & ev,const edm::EventSetup & es);

    void setTrackingRegion(const TrackingRegion * trackingRegion){trackingRegion_ = trackingRegion;}
    
    bool pass(const std::vector<const FastTrackerRecHit *>& hits) const;
    //new for Phase1
    SeedingLayerSetsBuilder::SeedingLayerId Layer_tuple(const FastTrackerRecHit * hit) const;

private:
    
    std::unique_ptr<HitTripletGeneratorFromPairAndLayers> pixelTripletGenerator_;
    std::unique_ptr<MultiHitGeneratorFromPairAndLayers> multiHitGenerator_;
    const TrackingRegion * trackingRegion_;
    const edm::EventSetup * eventSetup_;
    const MeasurementTracker * measurementTracker_;
    const std::string measurementTrackerLabel_;
    std::unique_ptr<CAHitTripletGenerator> CAHitTriplGenerator_;
    std::unique_ptr<CAHitQuadrupletGenerator> CAHitQuadGenerator_;    
    std::unique_ptr<SeedingLayerSetsBuilder> seedingLayers_;
    std::unique_ptr<SeedingLayerSetsHits> seedingLayer;
    std::vector<unsigned> layerPairs_;
    edm::ESHandle<TrackerTopology> trackerTopology;
    std::vector<SeedingLayerSetsBuilder::SeedingLayerId> seedingLayerIds;
};

#endif
