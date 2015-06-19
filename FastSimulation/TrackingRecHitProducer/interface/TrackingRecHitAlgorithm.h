#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

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
        const TrackerGeometry* _trackerGeometry;

    public:
        TrackingRecHitAlgorithm(const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector);
            
        inline void setupTopologyAndGeometry(const TrackerTopology* trackerTopology, const TrackerGeometry* trackerGeometry)
        {
            _trackerTopology=trackerTopology;
            _trackerGeometry=trackerGeometry;
        }
        
        inline const TrackerTopology* getTrackerTopology() const
        {
            return _trackerTopology;
        }

        inline const TrackerGeometry* getTrackerGeometry() const
        {
            return _trackerGeometry;
        }

        //this function will only be called once per event
        virtual void beginEvent(edm::Event& event, const edm::EventSetup& eventSetup);

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const;

        //this function will only be called once per event
        virtual void endEvent(edm::Event& event, const edm::EventSetup& eventSetup);

        virtual ~TrackingRecHitAlgorithm();
        
        inline std::string getSelectionString() const
        {
            return _selectionString;
        }

};


#endif
