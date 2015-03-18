#include "TrackingRecHitProducer.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"


#include <map>

TrackingRecHitProducer::TrackingRecHitProducer(const edm::ParameterSet& config):
    _trackerGeometry(nullptr),
    _trackerTopology(nullptr)
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

}

void TrackingRecHitProducer::beginRun(edm::Run const&, const edm::EventSetup& eventSetup)
{
    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<TrackerTopology> trackerTopologyHandle;
    eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    eventSetup.get<IdealGeometryRecord>().get(trackerTopologyHandle);
    _trackerGeometry = trackerGeometryHandle.product();
    _trackerTopology = trackerTopologyHandle.product();
    
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->setupTrackerTopology(_trackerTopology);
    }
    
    
    //build pipes for all detIds
    const std::vector<DetId>& detIds = _trackerGeometry->detIds();
    
    for (const DetId& detId: detIds)
    {
        TrackerDetIdSelector selector(detId,*_trackerTopology);
       
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

void TrackingRecHitProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
    //build DetId -> PSimHit map
    edm::Handle<std::vector<PSimHit>> simHits;
    event.getByToken(_simHitToken,simHits);
    std::map<unsigned int,std::vector<const PSimHit*>> hitsPerDetId;
    for (unsigned int ihit = 0; ihit < simHits->size(); ++ihit)
    {
        const PSimHit* simHit = &(*simHits)[ihit];
        hitsPerDetId[simHit->detUnitId()].push_back(simHit);
    }
    
    //run pipes
    for (std::map<unsigned int,std::vector<const PSimHit*>>::const_iterator simHitsIt = hitsPerDetId.cbegin(); simHitsIt != hitsPerDetId.cend(); ++simHitsIt)
    {
    
        TrackingRecHitProductPtr product = std::make_shared<TrackingRecHitProduct>(simHitsIt->first,simHitsIt->second);
        std::map<unsigned int, TrackingRecHitPipe>::const_iterator pipeIt = _detIdPipes.find(simHitsIt->first);
        if (pipeIt!=_detIdPipes.cend())
        {
            const TrackingRecHitPipe& pipe = pipeIt->second;
            pipe.produce(product);
        }
        else
        {
            //there should be at least an empty pipe for each DetId
            throw cms::Exception("FastSimulation/TrackingRecHitProducer","A PSimHit carries a DetId which does not belong to the TrackerGeometry: "+_trackerTopology->print(simHitsIt->first));
        }
    }
}

TrackingRecHitProducer::~TrackingRecHitProducer()
{
}


DEFINE_FWK_MODULE(TrackingRecHitProducer);
