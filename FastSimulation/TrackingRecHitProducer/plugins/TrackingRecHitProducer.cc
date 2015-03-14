#include "TrackingRecHitProducer.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#include <map>

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

    _selection = config.getParameter<std::string>("select");
}

void TrackingRecHitProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
    edm::Handle<std::vector<PSimHit>> simHits;
    event.getByToken(_simHitToken,simHits);
    std::map<unsigned int,std::vector<const PSimHit*>> hitsPerDetId;

    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<TrackerTopology> trackerTopologyHandle;
    eventSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    eventSetup.get<IdealGeometryRecord>().get(trackerTopologyHandle);

    const TrackerGeometry& trackerGeometry = *trackerGeometryHandle;
    const TrackerTopology& trackerTopology = *trackerTopologyHandle;

    for (unsigned int ihit = 0; ihit < simHits->size(); ++ihit)
    {
        const PSimHit* simHit = &(*simHits)[ihit];
        if (hitsPerDetId.find(simHit->detUnitId ())==hitsPerDetId.end())
        {
            const DetId detId(simHit->detUnitId());
            TrackerDetIdSelector selector(detId,trackerTopology);

            //std::cout<<&trackerTopology<<", "<<&detId<<": "<<trackerTopology.pxbLayer(detId)<<" | "<<trackerTopology.print(detId)<<std::endl;
            std::cout<<trackerTopology.print(detId)<<std::endl;
            bool selected = selector.passSelection(_selection);
            std::cout<<"selected="<<(selected ? "true" : "false")<<std::endl;
            std::cout<<std::endl;

            if (hitsPerDetId.size()>10)
            {
                break;
            }
        }
        hitsPerDetId[simHit->detUnitId ()].push_back(simHit);
    }
    for (std::map<unsigned int,std::vector<const PSimHit*>>::const_iterator it = hitsPerDetId.cbegin(); it != hitsPerDetId.cend(); ++it)
    {
        DetId detId(it->first);
        for (unsigned int ialgo = 0; ialgo <_recHitAlgorithms.size(); ++ialgo)
        {
            _recHitAlgorithms[ialgo]->processDetId(detId, trackerTopology, trackerGeometry, it->second);
        }
    }
}

TrackingRecHitProducer::~TrackingRecHitProducer()
{
}


DEFINE_FWK_MODULE(TrackingRecHitProducer);
