#include "TrackingRecHitProducer.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"


#include <map>
#include <memory>



void insertRecHits(
    TrackerGSRecHitCollection& targetCollection,
    TrackingRecHitProductPtr product
)
{
    if (product)
    {
        TrackerGSRecHitCollection::FastFiller filler(targetCollection,product->getDetId());
        std::vector<SiTrackerGSRecHit2D>& recHits = product->getRecHits();
        filler.resize(recHits.size());
        for (unsigned int ihit = 0; ihit < recHits.size(); ++ihit)
        {
            filler[ihit]=recHits[ihit];
        }
    }
}

void insertMatchedRecHits(
    TrackerGSMatchedRecHitCollection& targetCollection,
    TrackingRecHitProductPtr product
)
{
    if (product)
    {
        TrackerGSMatchedRecHitCollection::FastFiller filler(targetCollection,product->getDetId());
        std::vector<SiTrackerGSMatchedRecHit2D>& recHits = product->getMatchedRecHits();
        filler.resize(recHits.size());
        for (unsigned int ihit = 0; ihit < recHits.size(); ++ihit)
        {
            filler[ihit]=recHits[ihit];
        }
    }
}



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

    produces<TrackerGSRecHitCollection>("TrackerGSRecHits");
    produces<TrackerGSMatchedRecHitCollection>("TrackerGSMatchedRecHits");

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

    std::auto_ptr<TrackerGSRecHitCollection> recHitOutputCollection(new TrackerGSRecHitCollection());
    std::auto_ptr<TrackerGSMatchedRecHitCollection> matchedRecHitOutputCollection(new TrackerGSMatchedRecHitCollection());
    
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

            insertRecHits(*recHitOutputCollection,product);
            insertMatchedRecHits(*matchedRecHitOutputCollection,product);

        }
        else
        {
            //there should be at least an empty pipe for each DetId
            throw cms::Exception("FastSimulation/TrackingRecHitProducer","A PSimHit carries a DetId which does not belong to the TrackerGeometry: "+_trackerTopology->print(simHitsIt->first));
        }
    }

    event.put(recHitOutputCollection,"TrackerGSRecHits");
    event.put(matchedRecHitOutputCollection,"TrackerGSMatchedRecHits");
    
    //begin event
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->endEvent(event,eventSetup);
    }
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
