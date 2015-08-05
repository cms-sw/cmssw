#include "TrackingRecHitProducer.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"

#include "DataFormats/Common/interface/RefVector.h"

#include <map>
#include <memory>

TrackingRecHitProducer::TrackingRecHitProducer(const edm::ParameterSet& config)
{
    edm::ConsumesCollector consumeCollector = consumesCollector();
    const edm::ParameterSet& pluginConfigs = config.getParameter<edm::ParameterSet>("plugins");
    const std::vector<std::string> psetNames = pluginConfigs.getParameterNamesForType<edm::ParameterSet>();

    for (unsigned int iplugin = 0; iplugin<psetNames.size(); ++iplugin)
    {
        const edm::ParameterSet& pluginConfig = pluginConfigs.getParameter<edm::ParameterSet>(psetNames[iplugin]);
        const std::string pluginName = pluginConfig.getParameter<std::string>("type");
        TrackingRecHitAlgorithm* recHitAlgorithm = TrackingRecHitAlgorithmFactory::get()->tryToCreate(pluginName,psetNames[iplugin],pluginConfig,consumeCollector);
        if (recHitAlgorithm)
        {
            _recHitAlgorithms.push_back(recHitAlgorithm);
        }
        else
        {
            edm::LogWarning("TrackingRecHitAlgorithm plugin not found: ") << "plugin name = "<<pluginName<<"\nconfiguration=\n"<<pluginConfig.dump();
        }
    }

    edm::InputTag simHitTag = config.getParameter<edm::InputTag>("simHits");
    _simHitToken = consumes<std::vector<PSimHit>>(simHitTag);

    //produces<std::vector<SiTrackerGSRecHit2D>>("TrackerGSRecHits");
}

void TrackingRecHitProducer::beginStream(edm::StreamID id)
{
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->beginStream(id);
    }
}

void TrackingRecHitProducer::beginRun(edm::Run const&, const edm::EventSetup& eventSetup)
{

}

void TrackingRecHitProducer::setupDetIdPipes(const edm::EventSetup& eventSetup)
{
    if (_iovSyncValue!=eventSetup.iovSyncValue())
    {
        _iovSyncValue=eventSetup.iovSyncValue();
        edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
        edm::ESHandle<TrackerTopology> trackerTopologyHandle;
        eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
        eventSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
        const TrackerGeometry* trackerGeometry = trackerGeometryHandle.product();
        const TrackerTopology* trackerTopology = trackerTopologyHandle.product();

        _detIdPipes.clear();

        //build pipes for all detIds
        const std::vector<DetId>& detIds = trackerGeometry->detIds();

        for (const DetId& detId: detIds)
        {
            TrackerDetIdSelector selector(detId,*trackerTopology);

            TrackingRecHitPipe pipe;
            for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
            {
                if (selector.passSelection(algo->getSelectionString()))
                {
                    pipe.addAlgorithm(algo);
                }
            }
            _detIdPipes[detId.rawId()]=pipe;
        }
    }
}

void TrackingRecHitProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
    //resetup pipes if new iov
    setupDetIdPipes(eventSetup);
    //begin event
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->beginEvent(event,eventSetup);
    }

    //build DetId -> PSimHit map
    edm::Handle<std::vector<PSimHit>> simHits;
    event.getByToken(_simHitToken,simHits);
    std::map<unsigned int,std::vector<const PSimHit*>> hitsPerDetId;
    for (unsigned int ihit = 0; ihit < simHits->size(); ++ihit)
    {
        const PSimHit* simHit = &(*simHits)[ihit];
        hitsPerDetId[simHit->detUnitId()].push_back(simHit);
    }

    //std::auto_ptr<std::vector<SiTrackerGSRecHit2D>> recHitOutputCollection(new std::vector<SiTrackerGSRecHit2D>());
    //std::auto_ptr<edm::RefVector<std::vector<PSimHit>>> simHitRefOutputCollection(new edm::RefVector<std::vector<PSimHit>>());

    //run pipes
    for (std::map<unsigned int,std::vector<const PSimHit*>>::iterator simHitsIt = hitsPerDetId.begin(); simHitsIt != hitsPerDetId.end(); ++simHitsIt)
    {
        const DetId& detId = simHitsIt->first;
        std::map<unsigned int, TrackingRecHitPipe>::const_iterator pipeIt = _detIdPipes.find(detId);
        if (pipeIt!=_detIdPipes.cend())
        {
            std::vector<const PSimHit*>& simHits = simHitsIt->second;
            const TrackingRecHitPipe& pipe = pipeIt->second;

            TrackingRecHitProductPtr product = std::make_shared<TrackingRecHitProduct>(detId,simHits);

            product = pipe.produce(product);

        }
        else
        {
            //there should be at least an empty pipe for each DetId
            throw cms::Exception("FastSimulation/TrackingRecHitProducer","A PSimHit carries a DetId which does not belong to the TrackerGeometry: "+std::to_string(simHitsIt->first));
        }
    }
    
    //end event
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->endEvent(event,eventSetup);
    }

    //event.put(recHitOutputCollection,"TrackerGSRecHits");


}

void TrackingRecHitProducer::endStream()
{
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->endStream();
    }
}

TrackingRecHitProducer::~TrackingRecHitProducer()
{
}


DEFINE_FWK_MODULE(TrackingRecHitProducer);
