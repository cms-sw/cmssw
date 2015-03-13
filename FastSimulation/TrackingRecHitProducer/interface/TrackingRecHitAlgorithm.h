#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H

#include <vector>

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

        virtual std::vector<SiTrackerGSRecHit2D> processDetId(
            const DetId& detId,
            const TrackerTopology& trackerTopology,
            const TrackerGeometry& trackerGeometry,
            const std::vector<const PSimHit*>& simHits
        ) const;

        virtual void endEvent(edm::Event& event, edm::EventSetup& eventSetup);

        virtual ~TrackingRecHitAlgorithm();

};


#endif
