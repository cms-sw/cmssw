#include "FastSimulation/Tracking/interface/SeedFinderSelector.h"

// framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

// track reco
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayers.h"
#include "RecoTracker/TkSeedGenerator/interface/MultiHitGeneratorFromPairAndLayersFactory.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayersFactory.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CAHitTripletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CAHitQuadrupletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
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

    if(cfg.exists("CAHitTripletGeneratorFactory"))
    {
        const edm::ParameterSet & tripletConfig = cfg.getParameter<edm::ParameterSet>("CAHitTripletGeneratorFactory");
	CAHitTriplGenerator_ = std::make_unique<CAHitTripletGenerator>(tripletConfig,consumesCollector);
	seedingLayers_ = std::make_unique<SeedingLayerSetsBuilder>(cfg, consumesCollector, edm::InputTag("fastTrackerRecHits")); //calling the new FastSim specific constructor to make SeedingLayerSetsHits pointer for triplet iterations
        layerPairs_ = cfg.getParameter<std::vector<unsigned>>("layerPairs"); //allowed layer pairs for CA triplets
    }

    if(cfg.exists("CAHitQuadrupletGeneratorFactory"))
    {
        const edm::ParameterSet & quadrupletConfig = cfg.getParameter<edm::ParameterSet>("CAHitQuadrupletGeneratorFactory");
	CAHitQuadGenerator_ = std::make_unique<CAHitQuadrupletGenerator>(quadrupletConfig, consumesCollector);
	seedingLayers_ = std::make_unique<SeedingLayerSetsBuilder>(cfg, consumesCollector, edm::InputTag("fastTrackerRecHits")); //calling the new FastSim specific constructor to make SeedingLayerSetsHits pointer for quadruplet iterations
	layerPairs_ = cfg.getParameter<std::vector<unsigned>>("layerPairs"); //allowed layer pairs for CA quadruplets
    }

    if((pixelTripletGenerator_ && multiHitGenerator_) || (CAHitTriplGenerator_ && pixelTripletGenerator_) || (CAHitTriplGenerator_ && multiHitGenerator_))
      {
	throw cms::Exception("FastSimTracking") << "It is forbidden to specify together 'pixelTripletGeneratorFactory', 'CAHitTripletGeneratorFactory' and 'MultiHitGeneratorFactory' in configuration of SeedFinderSelection";
      }
    if((pixelTripletGenerator_ && CAHitQuadGenerator_) || (CAHitTriplGenerator_ && CAHitQuadGenerator_) || (multiHitGenerator_ && CAHitQuadGenerator_))
      {
	throw cms::Exception("FastSimTracking") << "It is forbidden to specify 'CAHitQuadrupletGeneratorFactory' together with 'pixelTripletGeneratorFactory', 'CAHitTripletGeneratorFactory' or 'MultiHitGeneratorFactory' in configuration of SeedFinderSelection";
      }  
}


SeedFinderSelector::~SeedFinderSelector(){;}

void SeedFinderSelector::initEvent(const edm::Event & ev,const edm::EventSetup & es)
{
    eventSetup_ = &es;
     
    edm::ESHandle<MeasurementTracker> measurementTrackerHandle;
    es.get<CkfComponentsRecord>().get(measurementTrackerLabel_, measurementTrackerHandle);
    es.get<TrackerTopologyRcd>().get(trackerTopology);
    measurementTracker_ = &(*measurementTrackerHandle);

    if(multiHitGenerator_)
    {
        multiHitGenerator_->initES(es);
    }

    //for CA triplet iterations
    if(CAHitTriplGenerator_){
      seedingLayer = seedingLayers_->makeSeedingLayerSetsHitsforFastSim(ev, es);
      seedingLayerIds = seedingLayers_->layers();
      CAHitTriplGenerator_->initEvent(ev,es);
    }
    //for CA quadruplet iterations
    if(CAHitQuadGenerator_){
      seedingLayer = seedingLayers_->makeSeedingLayerSetsHitsforFastSim(ev, es);
      seedingLayerIds = seedingLayers_->layers();
      CAHitQuadGenerator_->initEvent(ev,es);
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
    
    std::vector<BaseTrackerRecHit const *> firstHits{hits[0]};
    std::vector<BaseTrackerRecHit const *> secondHits{hits[1]};
    
    const RecHitsSortedInPhi fhm(firstHits, trackingRegion_->origin(), firstLayer);
    const RecHitsSortedInPhi shm(secondHits, trackingRegion_->origin(), secondLayer);
    
    HitDoublets result(fhm,shm);
    HitPairGeneratorFromLayerPair::doublets(*trackingRegion_,*firstLayer,*secondLayer,fhm,shm,*eventSetup_,0,result);
    
    if(result.empty())
    {
	return false;
    }
    
    // check the inner 3 hits
    if(pixelTripletGenerator_ || multiHitGenerator_ || CAHitTriplGenerator_)
    {
	if(hits.size() < 3)
	{
	    throw cms::Exception("FastSimTracking") << "For the given configuration, SeedFinderSelector::pass requires at least 3 hits";
	}
	const DetLayer * thirdLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[2]->det()->geographicalId());
	std::vector<const DetLayer *> thirdLayerDetLayer(1,thirdLayer);
	std::vector<BaseTrackerRecHit const *> thirdHits{hits[2]};
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
	//new for Phase1
	else if(CAHitTriplGenerator_)
	{  
	  if(!seedingLayer)
	    throw cms::Exception("FastSimTracking") << "ERROR: SeedingLayers pointer not set for CATripletGenerator";

	  SeedingLayerSetsHits & layers = *seedingLayer;
	  //constructing IntermediateHitDoublets to be passed onto CAHitTripletGenerator::hitNtuplets()
	  IntermediateHitDoublets ihd(&layers);
	  const TrackingRegion& tr_ = *trackingRegion_;
	  auto filler = ihd.beginRegion(&tr_); 

	  //Forming doublets for CA triplets from the allowed layer pair combinations:(0,1),(1,2)
	  std::array<SeedingLayerSetsBuilder::SeedingLayerId,2> hitPair;
	  for(int i=0; i<2; i++){
	    SeedingLayerSetsHits::SeedingLayerSet pairCandidate;
	    hitPair[0] = Layer_tuple(hits[i]);
	    hitPair[1] = Layer_tuple(hits[i+1]);

	    bool found;
	    for(SeedingLayerSetsHits::SeedingLayerSet ls : *seedingLayer){
	      found = false;
	      for(const auto p : layerPairs_){
		pairCandidate = ls.slice(p,p+2);
		if(hitPair[0] == seedingLayerIds[pairCandidate[0].index()] && hitPair[1] == seedingLayerIds[pairCandidate[1].index()]){
		  found = true;
		  break;
		}
	      }
	      if(found)
		break;
	    }
	    assert(found == true);
	    const DetLayer * fLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[i]->det()->geographicalId());
	    const DetLayer * sLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[i+1]->det()->geographicalId());
	    std::vector<BaseTrackerRecHit const *> fHits{hits[i]};
	    std::vector<BaseTrackerRecHit const *> sHits{hits[i+1]};
	    
	    //Important: doublets to be added to the cache
	    auto& layerCache = filler.layerHitMapCache();
	    const RecHitsSortedInPhi& firsthm = *layerCache.add(pairCandidate[0], std::make_unique<RecHitsSortedInPhi>(fHits, trackingRegion_->origin(),fLayer));
	    const RecHitsSortedInPhi& secondhm = *layerCache.add(pairCandidate[1], std::make_unique<RecHitsSortedInPhi>(sHits, trackingRegion_->origin(),sLayer));
	    HitDoublets res(firsthm,secondhm);
	    HitPairGeneratorFromLayerPair::doublets(*trackingRegion_,*fLayer,*sLayer,firsthm,secondhm,*eventSetup_,0,res);
	    filler.addDoublets(pairCandidate, std::move(res)); //fill the formed doublet
	  }
	  std::vector<OrderedHitSeeds> tripletresult;                                
	  tripletresult.resize(ihd.regionSize());
	  for(auto& ntuplet : tripletresult)
	    ntuplet.reserve(3);
	  CAHitTriplGenerator_->hitNtuplets(ihd,tripletresult,*eventSetup_,*seedingLayer); //calling the function from the class, modifies tripletresult
      	  return !tripletresult.empty();
	}
    }
    //new for Phase1     
    if(CAHitQuadGenerator_)
    {
      if(hits.size() < 4)
	{
	  throw cms::Exception("FastSimTracking") << "For the given configuration, SeedFinderSelector::pass requires at least 4 hits";
	}

      if(!seedingLayer)
	throw cms::Exception("FastSimTracking") << "ERROR: SeedingLayers pointer not set for CAHitQuadrupletGenerator";      

      SeedingLayerSetsHits & layers = *seedingLayer;
      //constructing IntermediateHitDoublets to be passed onto CAHitQuadrupletGenerator::hitNtuplets()
      IntermediateHitDoublets ihd(&layers);
      const TrackingRegion& tr_ = *trackingRegion_;
      auto filler = ihd.beginRegion(&tr_);
      
      //Forming doublets for CA quadruplets from the allowed layer pair combinations:(0,1),(1,2),(2,3)
      std::array<SeedingLayerSetsBuilder::SeedingLayerId,2> hitPair;
      for(int i=0; i<3; i++){
	SeedingLayerSetsHits::SeedingLayerSet pairCandidate;
	hitPair[0] = Layer_tuple(hits[i]);
 	hitPair[1] = Layer_tuple(hits[i+1]);
       
	bool found;
        for(SeedingLayerSetsHits::SeedingLayerSet ls : *seedingLayer){
	  found = false;
	  for(const auto p : layerPairs_){
	    pairCandidate = ls.slice(p,p+2);
	    if(hitPair[0] == seedingLayerIds[pairCandidate[0].index()] && hitPair[1] == seedingLayerIds[pairCandidate[1].index()]){
	      found = true;
	      break;
	    }
	  }
	  if(found)
	    break;
	}
	assert(found == true);
	const DetLayer * fLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[i]->det()->geographicalId());
	const DetLayer * sLayer = measurementTracker_->geometricSearchTracker()->detLayer(hits[i+1]->det()->geographicalId());
	std::vector<BaseTrackerRecHit const *> fHits{hits[i]};
	std::vector<BaseTrackerRecHit const *> sHits{hits[i+1]};

	//Important: doublets to be added to the cache
	auto& layerCache = filler.layerHitMapCache();
	const RecHitsSortedInPhi& firsthm = *layerCache.add(pairCandidate[0], std::make_unique<RecHitsSortedInPhi>(fHits, trackingRegion_->origin(), fLayer));
	const RecHitsSortedInPhi& secondhm = *layerCache.add(pairCandidate[1], std::make_unique<RecHitsSortedInPhi>(sHits, trackingRegion_->origin(), sLayer));
	HitDoublets res(firsthm,secondhm);
	HitPairGeneratorFromLayerPair::doublets(*trackingRegion_,*fLayer,*sLayer,firsthm,secondhm,*eventSetup_,0,res);
	filler.addDoublets(pairCandidate, std::move(res)); //fill the formed doublet
      }
      
      std::vector<OrderedHitSeeds> quadrupletresult;
      quadrupletresult.resize(ihd.regionSize());
      for(auto& ntuplet : quadrupletresult)
	ntuplet.reserve(4);
      CAHitQuadGenerator_->hitNtuplets(ihd,quadrupletresult,*eventSetup_,*seedingLayer); //calling the function from the class, modifies quadrupletresult
      return !quadrupletresult.empty();  
    }    

    return true;
    
}

//new for Phase1
SeedingLayerSetsBuilder::SeedingLayerId SeedFinderSelector::Layer_tuple(const FastTrackerRecHit * hit) const
{
  const TrackerTopology* const tTopo = trackerTopology.product();
  GeomDetEnumerators::SubDetector subdet = GeomDetEnumerators::invalidDet;
  TrackerDetSide side = TrackerDetSide::Barrel;
  int idLayer = 0;
  
  if( (hit->det()->geographicalId()).subdetId() == PixelSubdetector::PixelBarrel){
    subdet = GeomDetEnumerators::PixelBarrel;
    side = TrackerDetSide::Barrel;
    idLayer = tTopo->pxbLayer(hit->det()->geographicalId());
  }
  else if ((hit->det()->geographicalId()).subdetId() == PixelSubdetector::PixelEndcap){
    subdet = GeomDetEnumerators::PixelEndcap;
    idLayer = tTopo->pxfDisk(hit->det()->geographicalId());
    if(tTopo->pxfSide(hit->det()->geographicalId())==1){
      side = TrackerDetSide::NegEndcap;
    }
    else{
      side = TrackerDetSide::PosEndcap;
    }
  }
  return std::make_tuple(subdet, side, idLayer);
}
