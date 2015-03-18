#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include <string>

namespace edm
{
    class Event;
    class EventSetup;
    class ParameterSet;
    class ConsumesCollector;
}




class TrackingRecHitAlgorithm
{
    protected:
        std::string _selectionString;
        const TrackerTopology* _trackerTopology;
    public:
        TrackingRecHitAlgorithm(const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector);
            
        inline void setupTrackerTopology(const TrackerTopology* trackerTopology)
        {
            _trackerTopology=trackerTopology;
        }
        
        inline const TrackerTopology* getTrackerTopology() const
        {
            return _trackerTopology;
        }

        virtual void beginEvent(const edm::Event& event, const edm::EventSetup& eventSetup);

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const;

        virtual void endEvent(edm::Event& event, edm::EventSetup& eventSetup);

        virtual ~TrackingRecHitAlgorithm();
        
        inline std::string getSelectionString() const
        {
            return _selectionString;
        }

};


#endif
