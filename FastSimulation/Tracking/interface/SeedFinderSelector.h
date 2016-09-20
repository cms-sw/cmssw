#ifndef SEEDFINDERSELECTOR_H
#define SEEDFINDERSELECTOR_H

#include <vector>
#include <memory>
#include <string>

class TrackingRegion;
class FastTrackerRecHit;
class MultiHitGeneratorFromPairAndLayers;
class HitTripletGeneratorFromPairAndLayers;
class MeasurementTracker;

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

private:
    
    std::unique_ptr<HitTripletGeneratorFromPairAndLayers> pixelTripletGenerator_;
    std::unique_ptr<MultiHitGeneratorFromPairAndLayers> multiHitGenerator_;
    const TrackingRegion * trackingRegion_;
    const edm::EventSetup * eventSetup_;
    const MeasurementTracker * measurementTracker_;
    const std::string measurementTrackerLabel_;
    
};

#endif
