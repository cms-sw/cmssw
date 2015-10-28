#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"

template class SeedingTree<TrackingLayer>;
template class SeedingNode<TrackingLayer>;

TrajectorySeedProducer::TrajectorySeedProducer(const edm::ParameterSet& conf)
{

    // produces
    produces<TrajectorySeedCollection>();

    // consumes
    recHitCombinationsToken = consumes<FastTrackerRecHitCombinationCollection>(conf.getParameter<edm::InputTag>("recHitCombinations"));
    if (conf.exists("hitMasks")){
	hitMasksToken = consumes<std::vector<bool> >(conf.getParameter<edm::InputTag>("hitMasks"));
    }

    // read Layers
    std::vector<std::string> layerStringList = conf.getParameter<std::vector<std::string>>("layerList");
    for(auto it=layerStringList.cbegin(); it < layerStringList.cend(); ++it) 
    {
        std::vector<TrackingLayer> trackingLayerList;
        std::string line = *it;
        std::string::size_type pos=0;
        while (pos != std::string::npos) 
        {
            pos=line.find("+");
            std::string layer = line.substr(0, pos);
            TrackingLayer layerSpec = TrackingLayer::createFromString(layer);

            trackingLayerList.push_back(layerSpec);
            line=line.substr(pos+1,std::string::npos); 
        }
        _seedingTree.insert(trackingLayerList);
    }

    if(conf.exists("RegionFactoryPSet")){
	/// region producer
	edm::ParameterSet regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
	std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
	theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector()));
	
	// seed creator
	const edm::ParameterSet & seedCreatorPSet = conf.getParameter<edm::ParameterSet>("SeedCreatorPSet");
	std::string seedCreatorName = seedCreatorPSet.getParameter<std::string>("ComponentName");
	seedCreator.reset(SeedCreatorFactory::get()->create( seedCreatorName, seedCreatorPSet));
    }

    // other parameters
    measurementTrackerLabel = conf.getParameter<std::string>("measurementTracker");
    
}


void TrajectorySeedProducer::produce(edm::Event& e, const edm::EventSetup& es) 
{        

    // services
    edm::ESHandle<TrackerGeometry> trackerGeometry;
    edm::ESHandle<TrackerTopology> trackerTopology;
    edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
    
    es.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
    es.get<TrackerTopologyRcd>().get(trackerTopology);
    es.get<CkfComponentsRecord>().get(measurementTrackerLabel, measurementTrackerHandle);
    measurementTracker = &(*measurementTrackerHandle);
    
    es_ = &es;

    // hit masks
    const std::vector<bool> * hitMasks = 0;
    if (!hitMasksToken.isUninitialized())
    {
	    edm::Handle<std::vector<bool> > hitMasksHandle;
	    e.getByToken(hitMasksToken,hitMasksHandle);
	    hitMasks = &(*hitMasksHandle);
    }
    
    // hit combinations
    edm::Handle<FastTrackerRecHitCombinationCollection> recHitCombinations;
    e.getByToken(recHitCombinationsToken, recHitCombinations);

    // output
    std::unique_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());

    // produce the regions;
    if(!theRegionProducer)
    {
	    edm::LogWarning("TrajectorySeedProducer") << " RegionFactory is not initialised properly, please check your configuration. Producing empty seed collection" << std::endl;
	    e.put(std::move(output));
	    return;
    }
    
    regions = theRegionProducer->regions(e,es);
    
    SeedFinder seedFinder(_seedingTree);
    
    //lambda function
    SeedFinder::Selector selectorFunction = [&](const std::vector<const TrajectorySeedHitCandidate*>& hits) -> bool
    {
        if (hits.size()==2)
        {
            const TrajectorySeedHitCandidate& innerHit = *hits[0];
            const TrajectorySeedHitCandidate& outerHit = *hits[1];
            
            const DetLayer * innerLayer = measurementTracker->geometricSearchTracker()->detLayer(innerHit.hit()->det()->geographicalId());
            const DetLayer * outerLayer = measurementTracker->geometricSearchTracker()->detLayer(outerHit.hit()->det()->geographicalId());
          
            typedef PixelRecoRange<float> Range;

            for(const auto & region : regions){

	        auto const & gs = outerHit.hit()->globalState();
	        auto loc = gs.position-region->origin().basicVector();
	        const HitRZCompatibility * checkRZ = region->checkRZ(innerLayer, outerHit.hit(), *es_, outerLayer,
							            loc.perp(),gs.position.z(),gs.errorR,gs.errorZ);

	        float u = innerLayer->isBarrel() ? loc.perp() : gs.position.z();
	        float v = innerLayer->isBarrel() ? gs.position.z() : loc.perp();
	        float dv = innerLayer->isBarrel() ? gs.errorZ : gs.errorR;
	        constexpr float nSigmaRZ = 3.46410161514f;
	        Range allowed = checkRZ->range(u);
	        float vErr = nSigmaRZ * dv;
	        Range hitRZ(v-vErr, v+vErr);
	        Range crossRange = allowed.intersection(hitRZ);

	        if( ! crossRange.empty()){
	            seedCreator->init(*region,*es_,0);
	            return true;}

            }
            return false;
        }
        return true;
    };
    
    seedFinder.setHitSelector(selectorFunction);
    
    
    for ( unsigned icomb=0; icomb<recHitCombinations->size(); ++icomb)
	{
	    FastTrackerRecHitCombination recHitCombination = (*recHitCombinations)[icomb];

	    TrajectorySeedHitCandidate previousTrackerHit;
	    TrajectorySeedHitCandidate currentTrackerHit;

	    std::vector<TrajectorySeedHitCandidate> trackerRecHits;
	    for (const auto & _hit : recHitCombination )
		{
		    // skip masked hits
		    if(hitMasks && fastTrackingUtilities::hitIsMasked(_hit.get(),hitMasks))
		    {
			    continue;
		    }
		
		    previousTrackerHit=currentTrackerHit;
	  
		    currentTrackerHit = TrajectorySeedHitCandidate(_hit.get(),trackerGeometry.product(),trackerTopology.product());
	  
		    if (_seedingTree.getSingleSet().find(currentTrackerHit.getTrackingLayer())!=_seedingTree.getSingleSet().end())
			{
			    //add only the hits which are actually on the requested layers
			    trackerRecHits.push_back(std::move(currentTrackerHit));
			}
		}

	    // find the first combination of hits
        std::vector<unsigned int> seedHitNumbers = seedFinder.getSeed(trackerRecHits);
        
	    // create a seed from those hits
	    if (seedHitNumbers.size()>1)
	    {
		    // temporary hit copies to allow setting the recHitCombinationIndex
		
		    edm::OwnVector<FastTrackerRecHit> seedHits;
		    for(unsigned iIndex = 0;iIndex < seedHitNumbers.size();++iIndex)
		    {
		        seedHits.push_back(trackerRecHits[seedHitNumbers[iIndex]].hit()->clone());
		    }
		    fastTrackingUtilities::setRecHitCombinationIndex(seedHits,icomb);

		    // the actual seed creation
		    seedCreator->makeSeed(
		        *output,
		        SeedingHitSet(
		            &seedHits[0],
		            &seedHits[1],
		            seedHits.size() >=3 ? &seedHits[2] : nullptr,
	                seedHits.size() >=4 ? &seedHits[3] : nullptr
                )
            );
	    }
	}
    e.put(std::move(output));

}
