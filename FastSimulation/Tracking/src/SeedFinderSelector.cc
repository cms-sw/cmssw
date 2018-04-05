#include "FastSimulation/Tracking/interface/SeedFinderSelector.h"

// framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// track reco
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"

// data formats
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"

SeedFinderSelector::SeedFinderSelector(const edm::ParameterSet & cfg,edm::ConsumesCollector && consumesCollector)
    : trackingRegion_(nullptr)
    , eventSetup_(nullptr)
    , measurementTracker_(nullptr)
    , measurementTrackerLabel_(cfg.getParameter<std::string>("measurementTracker"))
{
    if(cfg.exists("pixelTripletGeneratorFactory"))
    {
        const edm::ParameterSet & tripletConfig = cfg.getParameter<edm::ParameterSet>("pixelTripletGeneratorFactory");
        pixelTripletGenerator_.reset(HitTripletGeneratorFromPairAndLayersFactory::get()->create(tripletConfig.getParameter<std::string>("ComponentName"),tripletConfig,consumesCollector));
    }

    if(cfg.exists("MultiHitGeneratorFactory"))
    {
        const edm::ParameterSet & tripletConfig = cfg.getParameter<edm::ParameterSet>("MultiHitGeneratorFactory");
        multiHitGenerator_.reset(MultiHitGeneratorFromPairAndLayersFactory::get()->create(tripletConfig.getParameter<std::string>("ComponentName"),tripletConfig));
    }

    if(pixelTripletGenerator_ && multiHitGenerator_)
    {
	throw cms::Exception("FastSimTracking") << "It is forbidden to specify both 'pixelTripletGeneratorFactory' and 'MultiHitGeneratorFactory' in configuration of SeedFinderSelection";
    }
}


SeedFinderSelector::~SeedFinderSelector(){;}

void SeedFinderSelector::initEvent(const edm::Event & ev,const edm::EventSetup & es)
{
    eventSetup_ = &es;
    
    edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
    es.get<CkfComponentsRecord>().get(measurementTrackerLabel_, measurementTrackerHandle);
    measurementTracker_ = &(*measurementTrackerHandle);

    if(multiHitGenerator_)
    {
        multiHitGenerator_->initES(es);
    }
}


bool SeedFinderSelector::pass(const std::vector<const FastTrackerRecHit *>& hits) const
{
    if(!measurementTracker_ || !eventSetup_)
    {
	throw cms::Exception("FastSimTracking") << "ERROR: event not initialized";
    }
    if(!trackingRegion_)
    {
	throw cms::Exception("FastSimTracking") << "ERROR: trackingRegion not set";
    }


    // check the inner 2 hits
    if(hits.size() < 2)
    {
	throw cms::Exception("FastSimTracking") << "SeedFinderSelector::pass requires at least 2 hits";
    }
    const DetLayer * firstLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[0]->det()->geographicalId());
    const DetLayer * secondLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[1]->det()->geographicalId());
    
    std::vector<BaseTrackerRecHit const *> firstHits(1,static_cast<const BaseTrackerRecHit*>(hits[0]));
    std::vector<BaseTrackerRecHit const *> secondHits(1,static_cast<const BaseTrackerRecHit*>(hits[1]));
    
    const RecHitsSortedInPhi fhm(firstHits, trackingRegion_->origin(), firstLayer);
    const RecHitsSortedInPhi shm(secondHits, trackingRegion_->origin(), secondLayer);
    
    HitDoublets result(fhm,shm);
    HitPairGeneratorFromLayerPair::doublets(*trackingRegion_,*firstLayer,*secondLayer,fhm,shm,*eventSetup_,0,result);
    
    if(result.empty())
    {
	return false;
    }
    
    // check the inner 3 hits
    if(pixelTripletGenerator_ || multiHitGenerator_)
    {
	if(hits.size() < 3)
	{
	    throw cms::Exception("FastSimTracking") << "For the given configuration, SeedFinderSelector::pass requires at least 3 hits";
	}
	const DetLayer * thirdLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[2]->det()->geographicalId());
	std::vector<const DetLayer *> thirdLayerDetLayer(1,thirdLayer);
	std::vector<BaseTrackerRecHit const *> thirdHits(1,static_cast<const BaseTrackerRecHit*>(hits[2]));
	const RecHitsSortedInPhi thm(thirdHits,trackingRegion_->origin(), thirdLayer);
	const RecHitsSortedInPhi * thmp =&thm;
	
	if(pixelTripletGenerator_)
	{
	    OrderedHitTriplets tripletresult;
	    pixelTripletGenerator_->hitTriplets(*trackingRegion_,tripletresult,*eventSetup_,result,&thmp,thirdLayerDetLayer,1);
	    return !tripletresult.empty();
	}
	else if(multiHitGenerator_)
	{
	    OrderedMultiHits  tripletresult;
	    multiHitGenerator_->hitTriplets(*trackingRegion_,tripletresult,*eventSetup_,result,&thmp,thirdLayerDetLayer,1);
	    return !tripletresult.empty();
	}

    }
    return true;
    
}
