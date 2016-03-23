#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitAlgorithm_H

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

//#include "CLHEP/Random/RandomEngine.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"


#include <string>
#include <memory>

namespace edm
{
    class Event;
    class EventSetup;
    class ParameterSet;
    class ConsumesCollector;
    class Stream;
}

class TrackingRecHitAlgorithm
{
    private:
        const std::string _name;
        const std::string _selectionString;
        const TrackerTopology* _trackerTopology;
        const TrackerGeometry* _trackerGeometry;
        std::shared_ptr<RandomEngineAndDistribution> _randomEngine;

    public:
        TrackingRecHitAlgorithm(const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector);
            
        inline const std::string& getName() const
        {
            return _name;
        }
        inline const std::string& getSelectionString() const
        {
            return _selectionString;
        }
        
        const TrackerTopology& getTrackerTopology() const;
        const TrackerGeometry& getTrackerGeometry() const;
        const RandomEngineAndDistribution& getRandomEngine() const;

        //this function will only be called once per stream
        virtual void beginStream(const edm::StreamID& id);
        
        //this function will only be called once per event
        virtual void beginEvent(edm::Event& event, const edm::EventSetup& eventSetup);

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const;

        //this function will only be called once per event
        virtual void endEvent(edm::Event& event, const edm::EventSetup& eventSetup);
        
        //this function will only be called once per stream
        virtual void endStream();

        virtual ~TrackingRecHitAlgorithm();
        

};


#endif
