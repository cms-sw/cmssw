#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitProducer_h
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"


namespace edm
{
    class ParameterSet;
    class Event;
    class EventSetup;
}


class TrackingRecHitProducer:
    public edm::stream::EDProducer<>
{
    public:
        TrackingRecHitProducer(const edm::ParameterSet& config);

        virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

        virtual ~TrackingRecHitProducer();
};

#endif
