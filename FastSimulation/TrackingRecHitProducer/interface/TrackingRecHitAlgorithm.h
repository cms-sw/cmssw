#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include <vector>
#include <memory>

namespace edm
{
    class Event;
    class EventSetup;
}

class PSimHit;
class SiTrackerGSRecHit2D;
class DetId;
class TrackerTopology;
class TrackerGeometry;

class TrackingRecHitAlgorithm
{
    public:
        TrackingRecHitAlgorithm();

        virtual void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const;

        virtual void endEvent(edm::Event& event, edm::EventSetup& eventSetup);

        virtual ~TrackingRecHitAlgorithm();

};


#endif
