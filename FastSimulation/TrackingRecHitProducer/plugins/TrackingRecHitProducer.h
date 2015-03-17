#ifndef FastSimulation_TrackingRecHitProducer_TrackingRecHitProducer_h
#define FastSimulation_TrackingRecHitProducer_TrackingRecHitProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitPipe.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"

#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

namespace edm
{
    class ParameterSet;
    class Event;
    class EventSetup;
}

class TrackingRecHitAlgorithm;

class TrackingRecHitProducer:
    public edm::stream::EDProducer<>
{
    private:
        edm::EDGetTokenT<std::vector<PSimHit>> _simHitToken;

        std::vector<TrackingRecHitAlgorithm*> _recHitAlgorithms;

        std::string _selection;
        
        const TrackerGeometry* _trackerGeometry;
        const TrackerTopology* _trackerTopology;
        
        std::map<unsigned int, TrackingRecHitPipe> _detIdPipes; 

    public:
        TrackingRecHitProducer(const edm::ParameterSet& config);
        
        virtual void beginJob(const edm::EventSetup& eventSetup);

        virtual void produce(edm::Event& event, const edm::EventSetup& eventSetup);

        virtual ~TrackingRecHitProducer();
};

#endif
