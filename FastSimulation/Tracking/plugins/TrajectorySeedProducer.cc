/*!

  NOTE: what is called here 'FastTrackPreCandidate' is currently still known as 'FastTrackerRecHitCombination'
  
  TrajectorySeedProducer emulates the reconstruction of TrajectorySeeds in FastSim.

  The input data is a list of FastTrackPreCandidates.
  (input data are configured through the parameter 'src')
  In each given FastTrackPreCandidate,
  TrajectorySeedProducer searches for one combination of hits that matches all given seed requirements,
  (see parameters 'layerList','regionFactoryPSet')
  In that process it respects the order of the hits in the FastTrackPreCandidate.
  When such combination is found, a TrajectorySeed is reconstructed.
  Optionally, one can specify a list of hits to be ignored by TrajectorySeedProducer.
  (see parameter 'hitMasks')
  
  The output data is the list of reconstructed TrajectorySeeds.

*/

// system
#include <memory>
#include <vector>
#include <string>

// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

// data formats 
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

// reco track classes
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

// geometry
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

// fastsim
#include "FastSimulation/Tracking/interface/SeedingTree.h"
#include "FastSimulation/Tracking/interface/TrackingLayer.h"
#include "FastSimulation/Tracking/interface/FastTrackingUtilities.h"
#include "FastSimulation/Tracking/interface/SeedFinder.h"

class TrajectorySeedProducer
    : public edm::stream::EDProducer<>
{
private:
    
    // tokens

    edm::EDGetTokenT<FastTrackerRecHitCombinationCollection> recHitCombinationsToken;
    edm::EDGetTokenT<std::vector<bool> > hitMasksToken;       
    
    // other data members

    std::vector<std::vector<TrackingLayer>> seedingLayers;
    SeedingTree<TrackingLayer> _seedingTree; 

    std::unique_ptr<SeedCreator> seedCreator;
    std::unique_ptr<TrackingRegionProducer> theRegionProducer;
    std::string measurementTrackerLabel;

    bool skipSeedFinderSelector;

public:

    TrajectorySeedProducer(const edm::ParameterSet& conf);
    
    virtual void produce(edm::Event& e, const edm::EventSetup& es);

};

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

    /// region producer
    edm::ParameterSet regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
    std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
    theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector()));
    
    // seed creator
    const edm::ParameterSet & seedCreatorPSet = conf.getParameter<edm::ParameterSet>("SeedCreatorPSet");
    std::string seedCreatorName = seedCreatorPSet.getParameter<std::string>("ComponentName");
    seedCreator.reset(SeedCreatorFactory::get()->create( seedCreatorName, seedCreatorPSet));

    // other parameters
    measurementTrackerLabel = conf.getParameter<std::string>("measurementTracker");
    skipSeedFinderSelector = conf.getUntrackedParameter<bool>("skipSeedFinderSelector",false);

}


void TrajectorySeedProducer::produce(edm::Event& e, const edm::EventSetup& es) 
{        

    // services
    edm::ESHandle<TrackerTopology> trackerTopology;
    edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
    
    es.get<TrackerTopologyRcd>().get(trackerTopology);
    es.get<CkfComponentsRecord>().get(measurementTrackerLabel, measurementTrackerHandle);
    const MeasurementTracker * measurementTracker = &(*measurementTrackerHandle);
    
    // input data
    edm::Handle<FastTrackerRecHitCombinationCollection> recHitCombinations;
    e.getByToken(recHitCombinationsToken, recHitCombinations);
    const std::vector<bool> * hitMasks = 0;
    if (!hitMasksToken.isUninitialized())
    {
	edm::Handle<std::vector<bool> > hitMasksHandle;
	e.getByToken(hitMasksToken,hitMasksHandle);
	hitMasks = &(*hitMasksHandle);
    }
    
    // output data
    std::unique_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());

    // produce the regions;
    const auto regions = theRegionProducer->regions(e,es);
    // and make sure there is at least one region
    if(regions.size() == 0)
    {
	e.put(std::move(output));
	return;
    }
    
    // pointer to selected region
    TrackingRegion * selectedTrackingRegion = 0;
       
    // define a lambda function
    // to select hit pairs, triplets, ... compatible with the region
    SeedFinder::Selector selectorFunction = [&es,&measurementTracker,&selectedTrackingRegion](const std::vector<const FastTrackerRecHit*>& hits) -> bool
    {
	// criteria for hit pairs
	// based on HitPairGeneratorFromLayerPair::doublets( const TrackingRegion& region, const edm::Event & iEvent, const edm::EventSetup& iSetup, Layers layers)
	if (hits.size()==2)
	{
	    const FastTrackerRecHit * innerHit = hits[0];
	    const FastTrackerRecHit * outerHit = hits[1];
	    
	    const DetLayer * innerLayer = measurementTracker->geometricSearchTracker()->detLayer(innerHit->det()->geographicalId());
	    const DetLayer * outerLayer = measurementTracker->geometricSearchTracker()->detLayer(outerHit->det()->geographicalId());
	    
	    std::vector<BaseTrackerRecHit const *> innerHits(1,(const BaseTrackerRecHit*) innerHit->hit());
	    std::vector<BaseTrackerRecHit const *> outerHits(1,(const BaseTrackerRecHit*) outerHit->hit());
		
	    //const RecHitsSortedInPhi* ihm=new RecHitsSortedInPhi (innerHits, selectedTrackingRegion->origin(), innerLayer);
	    //const RecHitsSortedInPhi* ohm=new RecHitsSortedInPhi (outerHits, selectedTrackingRegion->origin(), outerLayer);
	    const RecHitsSortedInPhi ihm(innerHits, selectedTrackingRegion->origin(), innerLayer);
	    const RecHitsSortedInPhi ohm(outerHits, selectedTrackingRegion->origin(), outerLayer);
		
	    HitDoublets result(ihm,ohm); 
	    HitPairGeneratorFromLayerPair::doublets(*selectedTrackingRegion,*innerLayer,*outerLayer,ihm,ohm,es,0,result);
	    
	    if(result.size()!=0)
	    {
		return true; 
	    }
	    else
	    {
		return false;
	    }
	}
	else
	{
	
	    // no criteria for hit combinations that are not pairs
	    return true;
	}
    };

    if(skipSeedFinderSelector)
    {
	selectedTrackingRegion = regions[0].get();
	selectorFunction = [](const std::vector<const FastTrackerRecHit*>& hits) -> bool
	{
	    return true;
	};
    }


    // instantiate the seed finder
    SeedFinder seedFinder(_seedingTree,*trackerTopology.product());
    seedFinder.setHitSelector(selectorFunction);
    
    // loop over the combinations
    for ( unsigned icomb=0; icomb<recHitCombinations->size(); ++icomb)
    {
	FastTrackerRecHitCombination recHitCombination = (*recHitCombinations)[icomb];
		
	// create a list of hits cleaned from masked hits
	std::vector<const FastTrackerRecHit * > seedHitCandidates;
	for (const auto & _hit : recHitCombination )
	{
	    if(hitMasks && fastTrackingUtilities::hitIsMasked(_hit.get(),hitMasks))
	    {
		continue;
	    }
	    seedHitCandidates.push_back(_hit.get());
	}

	// loop over the regions
	for(auto region = regions.begin();region != regions.end(); ++region)
	{
	    
	    // set the region
	    selectedTrackingRegion = region->get();

	    // find the hits on the seeds
	    std::vector<unsigned int> seedHitNumbers = seedFinder.getSeed(seedHitCandidates);

	    // create a seed from those hits
	    if (seedHitNumbers.size()>1)
	    {

		// copy the hits and make them aware of the combination they originate from
		edm::OwnVector<FastTrackerRecHit> seedHits;
		for(unsigned iIndex = 0;iIndex < seedHitNumbers.size();++iIndex)
		{
		    seedHits.push_back(seedHitCandidates[seedHitNumbers[iIndex]]->clone());
		}
		fastTrackingUtilities::setRecHitCombinationIndex(seedHits,icomb);
	    
		seedCreator->init(*selectedTrackingRegion,es,0);
		seedCreator->makeSeed(
		    *output,
		    SeedingHitSet(
			&seedHits[0],
			&seedHits[1],
			seedHits.size() >=3 ? &seedHits[2] : nullptr,
			seedHits.size() >=4 ? &seedHits[3] : nullptr
			)
		    );
		break; // break the loop over the regions
	    }
	}
    }
    e.put(std::move(output));

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrajectorySeedProducer);
