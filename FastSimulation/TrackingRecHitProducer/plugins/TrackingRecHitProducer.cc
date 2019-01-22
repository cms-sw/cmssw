#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitPipe.h"
#include "FastSimulation/TrackingRecHitProducer/interface/PixelTemplateSmearerBase.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
// Pixel-related stuff:
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackerDetIdSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"

#include <map>
#include <memory>
#include <vector>

class TrackingRecHitProducer:
    public edm::stream::EDProducer<>
{
    private:
        edm::EDGetTokenT<std::vector<PSimHit>> _simHitToken;
        std::vector<TrackingRecHitAlgorithm*> _recHitAlgorithms;
        unsigned long long _trackerGeometryCacheID = 0;
        unsigned long long _trackerTopologyCacheID = 0;
        std::map<unsigned int, TrackingRecHitPipe> _detIdPipes;
        void setupDetIdPipes(const edm::EventSetup& eventSetup);
        std::vector< SiPixelTemplateStore > _pixelTempStore ;   // pixel template storage

    public:
        TrackingRecHitProducer(const edm::ParameterSet& config);

        void beginRun(edm::Run const&, const edm::EventSetup& eventSetup) override;

        void beginStream(edm::StreamID id) override;

        void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

        void endStream() override;

        ~TrackingRecHitProducer() override;
};


TrackingRecHitProducer::TrackingRecHitProducer(const edm::ParameterSet& config)
{
    edm::ConsumesCollector consumeCollector = consumesCollector();
    const std::vector<edm::ParameterSet>& pluginConfigs = config.getParameter<std::vector<edm::ParameterSet>>("plugins");

    for (unsigned int iplugin = 0; iplugin<pluginConfigs.size(); ++iplugin)
    {
        const edm::ParameterSet& pluginConfig = pluginConfigs[iplugin];
        const std::string pluginType = pluginConfig.getParameter<std::string>("type");
        const std::string pluginName = pluginConfig.getParameter<std::string>("name");

        TrackingRecHitAlgorithm* recHitAlgorithm = TrackingRecHitAlgorithmFactory::get()->tryToCreate(pluginType,pluginName,pluginConfig,consumeCollector);
        if (recHitAlgorithm)
        {
            edm::LogInfo("TrackingRecHitProducer: ")<< "adding plugin type '"<<pluginType<<"' as '"<<pluginName<<"'"<<std::endl;
            _recHitAlgorithms.push_back(recHitAlgorithm);
        }
        else
        {
            throw cms::Exception("TrackingRecHitAlgorithm plugin not found: ") << "plugin type = "<<pluginType<<"\nconfiguration=\n"<<pluginConfig.dump();
        }
    }

    edm::InputTag simHitTag = config.getParameter<edm::InputTag>("simHits");
    _simHitToken = consumes<std::vector<PSimHit>>(simHitTag);

    produces<FastTrackerRecHitCollection>();
    produces<FastTrackerRecHitRefCollection>("simHit2RecHitMap");
}

TrackingRecHitProducer::~TrackingRecHitProducer()
{
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        delete algo;
    }
    _recHitAlgorithms.clear();

    //--- Delete the templates. This is safe even if thePixelTemp_ vector is empty.
    for (auto x : _pixelTempStore) x.destroy();
}


void TrackingRecHitProducer::beginStream(edm::StreamID id)
{
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->beginStream(id);
    }
}

void TrackingRecHitProducer::beginRun(edm::Run const& run, const edm::EventSetup& eventSetup)
{
    //--- Since all pixel algorithms (of which there could be several) use the same
    //    templateStore, filled out from the same DB Object, we need to it centrally
    //    (namely here), and then distribute it to the algorithms.  Note that only
    //    the pixel algorithms implement beginRun(), for the strip tracker this defaults
    //    to a no-op.

    edm::ESHandle<SiPixelTemplateDBObject> templateDBobject;
    eventSetup.get<SiPixelTemplateDBObjectESProducerRcd>().get(templateDBobject);
    const SiPixelTemplateDBObject * pixelTemplateDBObject = templateDBobject.product();
  
    //--- Now that we have the DB object, load the correct templates from the DB.  
    //    (They are needed for data and full sim MC, so in a production FastSim
    //    run, everything should already be in the DB.)
    if ( !SiPixelTemplate::pushfile( *pixelTemplateDBObject, _pixelTempStore ) ) {
         throw cms::Exception("TrackingRecHitProducer:")
	   << "SiPixel Templates not loaded correctly from the DB object!" << std::endl;
    }

    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
      algo->beginRun(run, eventSetup, pixelTemplateDBObject, _pixelTempStore );
    }
}

void TrackingRecHitProducer::setupDetIdPipes(const edm::EventSetup& eventSetup)
{
    auto const& trackerGeomRec = eventSetup.get<TrackerDigiGeometryRecord>();
    auto const& trackerTopoRec = eventSetup.get<TrackerTopologyRcd>();
    if (trackerGeomRec.cacheIdentifier() != _trackerGeometryCacheID or 
        trackerTopoRec.cacheIdentifier() != _trackerTopologyCacheID )
    {
        _trackerGeometryCacheID = trackerGeomRec.cacheIdentifier();
        _trackerTopologyCacheID = trackerTopoRec.cacheIdentifier();
        edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
        edm::ESHandle<TrackerTopology> trackerTopologyHandle;
        trackerGeomRec.get(trackerGeometryHandle);
        trackerTopoRec.get(trackerTopologyHandle);
        const TrackerGeometry* trackerGeometry = trackerGeometryHandle.product();
        const TrackerTopology* trackerTopology = trackerTopologyHandle.product();

        _detIdPipes.clear();

        //build pipes for all detIds
        const std::vector<DetId>& detIds = trackerGeometry->detIds();
        std::vector<unsigned int> numberOfDetIdsPerAlgorithm(_recHitAlgorithms.size(),0);

        for (const DetId& detId: detIds)
        {
            TrackerDetIdSelector selector(detId,*trackerTopology);

            TrackingRecHitPipe pipe;
            for (unsigned int ialgo = 0; ialgo < _recHitAlgorithms.size(); ++ialgo)
            {
                TrackingRecHitAlgorithm* algo = _recHitAlgorithms[ialgo];
                if (selector.passSelection(algo->getSelectionString()))
                {
                    numberOfDetIdsPerAlgorithm[ialgo]+=1;
                    pipe.addAlgorithm(algo);
                }
            }
            if (pipe.size()==0)
            {
                throw cms::Exception("FastSimulation/TrackingRecHitProducer: DetId not configured! ("+trackerTopology->print(detId)+")");
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

    std::unique_ptr<FastTrackerRecHitCollection> output_recHits(new FastTrackerRecHitCollection);
    output_recHits->reserve(simHits->size());

    edm::RefProd<FastTrackerRecHitCollection> output_recHits_refProd = event.getRefBeforePut<FastTrackerRecHitCollection>();
    std::unique_ptr<FastTrackerRecHitRefCollection> output_recHitRefs(new FastTrackerRecHitRefCollection(simHits->size(),FastTrackerRecHitRef()));

    std::map<unsigned int,std::vector<std::pair<unsigned int,const PSimHit*>>> simHitsIdPairPerDetId;
    for (unsigned int ihit = 0; ihit < simHits->size(); ++ihit)
    {
        const PSimHit* simHit = &(*simHits)[ihit];
        simHitsIdPairPerDetId[simHit->detUnitId()].push_back(std::make_pair(ihit,simHit));
    }

    for (auto simHitsIdPairIt = simHitsIdPairPerDetId.begin(); simHitsIdPairIt != simHitsIdPairPerDetId.end(); ++simHitsIdPairIt)
    {
        const DetId& detId = simHitsIdPairIt->first;
        std::map<unsigned int, TrackingRecHitPipe>::const_iterator pipeIt = _detIdPipes.find(detId);
        if (pipeIt!=_detIdPipes.cend())
        {
            auto& simHitIdPairList = simHitsIdPairIt->second;
            const TrackingRecHitPipe& pipe = pipeIt->second;

            TrackingRecHitProductPtr product = std::make_shared<TrackingRecHitProduct>(detId,simHitIdPairList);

            product = pipe.produce(product);

            const std::vector<TrackingRecHitProduct::RecHitToSimHitIdPairs>& recHitToSimHitIdPairsList = product->getRecHitToSimHitIdPairs();


            for (unsigned int irecHit = 0; irecHit < recHitToSimHitIdPairsList.size(); ++irecHit)
            {
                output_recHits->push_back(recHitToSimHitIdPairsList[irecHit].first);
                const std::vector<TrackingRecHitProduct::SimHitIdPair>& simHitIdPairList = recHitToSimHitIdPairsList[irecHit].second;
                for (unsigned int isimHit = 0; isimHit < simHitIdPairList.size(); ++isimHit)
                {
                    unsigned int simHitId = simHitIdPairList[isimHit].first;
                    if (not (*output_recHitRefs)[simHitId].isNull())
                    {
                        throw cms::Exception("FastSimulation/TrackingRecHitProducer","A PSimHit cannot lead to multiple FastTrackerRecHits");
                    }
                    (*output_recHitRefs)[simHitId] = FastTrackerRecHitRef(output_recHits_refProd,output_recHits->size()-1);
                }
            }
        }
        else
        {
            //there should be at least an empty pipe for each DetId
            throw cms::Exception("FastSimulation/TrackingRecHitProducer","A PSimHit carries a DetId which does not belong to the TrackerGeometry: "+std::to_string(detId));
        }
    }

    //end event
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->endEvent(event,eventSetup);
    }

    // note from lukas:
    // all rechits need a unique id numbers
    for(unsigned recHitIndex = 0; recHitIndex < output_recHits->size(); ++recHitIndex)
    {
	    ((FastSingleTrackerRecHit*)&(*output_recHits)[recHitIndex])->setId(recHitIndex);
    }

    event.put(std::move(output_recHits));
    event.put(std::move(output_recHitRefs),"simHit2RecHitMap");

}

void TrackingRecHitProducer::endStream()
{
    for (TrackingRecHitAlgorithm* algo: _recHitAlgorithms)
    {
        algo->endStream();
    }
}

DEFINE_FWK_MODULE(TrackingRecHitProducer);
