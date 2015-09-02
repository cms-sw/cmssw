#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMapRecord.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "FastSimulation/Tracking/interface/HitMaskHelper.h"
#include "FastSimulation/Tracking/interface/FastTrackingHelper.h"

#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"

//Propagator withMaterial
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"


#include <unordered_set>

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

template class SeedingTree<TrackingLayer>;
template class SeedingNode<TrackingLayer>;

TrajectorySeedProducer::TrajectorySeedProducer(const edm::ParameterSet& conf):
    magneticField(nullptr),
    magneticFieldMap(nullptr),
    trackerGeometry(nullptr),
    trackerTopology(nullptr),
    theRegionProducer(nullptr)
{

    produces<TrajectorySeedCollection>();

    // tokens
    recHitCombinationsToken = consumes<FastTrackerRecHitCombinationCollection>(conf.getParameter<edm::InputTag>("recHitCombinations"));
    measurementTrackerEventToken = consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("MeasurementTrackerEvent"));
    hitMasks_exists = conf.exists("hitMasks");
    if (hitMasks_exists){
      edm::InputTag hitMasksTag = conf.getParameter<edm::InputTag>("hitMasks");   
      hitMasksToken = consumes<std::vector<bool> >(hitMasksTag);
    }
    
    // The smallest number of hits for a track candidate
    minLayersCrossed = conf.getParameter<unsigned int>("minLayersCrossed");

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
        seedingLayers.push_back(std::move(trackingLayerList));
    }

    // region producer
    if(conf.exists("RegionFactoryPSet")){
	edm::ParameterSet regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
	std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
	theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector()));
	
	// seed creator
	const edm::ParameterSet & seedCreatorPSet = conf.getParameter<edm::ParameterSet>("SeedCreatorPSet");
	std::string seedCreatorName = seedCreatorPSet.getParameter<std::string>("ComponentName");
	seedCreator.reset(SeedCreatorFactory::get()->create( seedCreatorName, seedCreatorPSet));
    }
}

bool
TrajectorySeedProducer::pass2HitsCuts(const TrajectorySeedHitCandidate & innerHit,const TrajectorySeedHitCandidate & outerHit) const
{

    const DetLayer * innerLayer = 
	measurementTrackerEvent->measurementTracker().geometricSearchTracker()->detLayer(innerHit.hit()->det()->geographicalId());
    const DetLayer * outerLayer = 
	measurementTrackerEvent->measurementTracker().geometricSearchTracker()->detLayer(outerHit.hit()->det()->geographicalId());
  
    typedef PixelRecoRange<float> Range;

    for(Regions::const_iterator ir=regions.begin(); ir < regions.end(); ++ir){

	auto const & gs = outerHit.hit()->globalState();
	auto loc = gs.position-(*ir)->origin().basicVector();
	const HitRZCompatibility * checkRZ = (*ir)->checkRZ(innerLayer, outerHit.hit(), *es_, outerLayer,
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
	    seedCreator->init(**ir,*es_,0);
	    return true;}

    }
    return false;
}

const SeedingNode<TrackingLayer>* TrajectorySeedProducer::insertHit(
								    const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
								    std::vector<int>& hitIndicesInTree,
								    const SeedingNode<TrackingLayer>* node, unsigned int trackerHit
								    ) const
{
  if (!node->getParent() || hitIndicesInTree[node->getParent()->getIndex()]>=0)
    {
      if (hitIndicesInTree[node->getIndex()]<0)
        {
	  const TrajectorySeedHitCandidate& currentTrackerHit = trackerRecHits[trackerHit];
	  if (!isHitOnLayer(currentTrackerHit,node->getData()))
            {
	      return nullptr;
            }
	  if (!passHitTuplesCuts(*node,trackerRecHits,hitIndicesInTree,currentTrackerHit))
            {
	      return nullptr;
            }
	  hitIndicesInTree[node->getIndex()]=trackerHit;
	  if (node->getChildrenSize()==0)
            {
	      return node;
            }
	  return nullptr;
        }
      else
        {
	  for (unsigned int ichild = 0; ichild<node->getChildrenSize(); ++ichild)
            {
	      const SeedingNode<TrackingLayer>* seed = insertHit(trackerRecHits,hitIndicesInTree,node->getChild(ichild),trackerHit);
	      if (seed)
                {
		  return seed;
                }
            }
        }
    }
    return nullptr;
}


std::vector<unsigned int> TrajectorySeedProducer::iterateHits(
							      unsigned int start,
							      const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
							      std::vector<int> hitIndicesInTree,
							      bool processSkippedHits
							      ) const
{
  for (unsigned int irecHit = start; irecHit<trackerRecHits.size(); ++irecHit)
    {
      unsigned int currentHitIndex=irecHit;
      
      for (unsigned int inext=currentHitIndex+1; inext< trackerRecHits.size(); ++inext)
        {
	  //if multiple hits are on the same layer -> follow all possibilities by recusion
	  if (trackerRecHits[currentHitIndex].getTrackingLayer()==trackerRecHits[inext].getTrackingLayer())
	    {
	      if (processSkippedHits)
		{
		  //recusively call the method again with hit 'inext' but skip all following on the same layer though 'processSkippedHits=false'
		  std::vector<unsigned int> seedHits = iterateHits(
								   inext,
								   trackerRecHits,
								   hitIndicesInTree,
								   false
								   );
		  if (seedHits.size()>0)
		    {
		      return seedHits;
                    }
                }
	      irecHit+=1; 
            }
	  else
	    {
	      break;
	    }
        }
      
      processSkippedHits=true;
      
      const SeedingNode<TrackingLayer>* seedNode = nullptr;
      for (unsigned int iroot=0; seedNode==nullptr && iroot<_seedingTree.numberOfRoots(); ++iroot)
	{
	  seedNode=insertHit(trackerRecHits,hitIndicesInTree,_seedingTree.getRoot(iroot), currentHitIndex);
	}
      if (seedNode)
	{
	  std::vector<unsigned int> seedIndices(seedNode->getDepth()+1);
	  while (seedNode)
	    {
	      seedIndices[seedNode->getDepth()]=hitIndicesInTree[seedNode->getIndex()];
	      seedNode=seedNode->getParent();
	    }
	  return seedIndices;
	}
	
    }
  
  return std::vector<unsigned int>();
  
}

void 
    TrajectorySeedProducer::produce(edm::Event& e, const edm::EventSetup& es) 
{        

    // services
    edm::ESHandle<MagneticField> magneticFieldHandle;
    edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
    edm::ESHandle<MagneticFieldMap> magneticFieldMapHandle;
    edm::ESHandle<TrackerTopology> trackerTopologyHandle;

    es.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
    es.get<TrackerDigiGeometryRecord>().get(trackerGeometryHandle);
    es.get<MagneticFieldMapRecord>().get(magneticFieldMapHandle);
    es.get<TrackerTopologyRcd>().get(trackerTopologyHandle);
    
    magneticField = &(*magneticFieldHandle);
    trackerGeometry = &(*trackerGeometryHandle);
    magneticFieldMap = &(*magneticFieldMapHandle);
    trackerTopology = &(*trackerTopologyHandle);

    es_ = &es;

    // hit masks
    std::unique_ptr<HitMaskHelper> hitMaskHelper;
    if (hitMasks_exists){  
	edm::Handle<std::vector<bool> > hitMasks;
	e.getByToken(hitMasksToken,hitMasks);
	hitMaskHelper.reset(new HitMaskHelper(hitMasks.product()));
    }
    
    // hit combinations
    edm::Handle<FastTrackerRecHitCombinationCollection> recHitCombinations;
    e.getByToken(recHitCombinationsToken, recHitCombinations);

    // measurement tracker event (only used to access services)
    edm::Handle<MeasurementTrackerEvent> measurementTrackerEventHandle;
    e.getByToken(measurementTrackerEventToken,measurementTrackerEventHandle);
    measurementTrackerEvent = measurementTrackerEventHandle.product();

    // output
    std::unique_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());

    // produce the regions;
    if(!theRegionProducer){
	edm::LogWarning("TrajectorySeedProducer") << " RegionFactory is not initialised properly, please check your configuration. Producing empty seed collection" << std::endl;
	e.put(std::move(output));
	return;
    }
    
    regions = theRegionProducer->regions(e,es);
    
    
    for ( unsigned icomb=0; icomb<recHitCombinations->size(); ++icomb)
	{
	  

	    FastTrackerRecHitCombination recHitCombination = (*recHitCombinations)[icomb];

	    TrajectorySeedHitCandidate previousTrackerHit;
	    TrajectorySeedHitCandidate currentTrackerHit;
	    unsigned int layersCrossed=0;
	

	    std::vector<TrajectorySeedHitCandidate> trackerRecHits;
	    for (const auto & _hit : recHitCombination )
		{
		    // skip masked hits
		    if(hitMaskHelper && hitMaskHelper->mask(_hit.get()))
			continue;
		
		    previousTrackerHit=currentTrackerHit;
	  
		    currentTrackerHit = TrajectorySeedHitCandidate(_hit.get(),trackerGeometry,trackerTopology);
	  
		    if (!currentTrackerHit.isOnTheSameLayer(previousTrackerHit))
			{
			    ++layersCrossed;
			}
		    if (_seedingTree.getSingleSet().find(currentTrackerHit.getTrackingLayer())!=_seedingTree.getSingleSet().end())
			{
			    //add only the hits which are actually on the requested layers
			    trackerRecHits.push_back(std::move(currentTrackerHit));
			}
		}
      
	    if ( layersCrossed < minLayersCrossed)
		{
		    continue;
		}

	    // set the combination index
      
	    //A SeedingNode is associated by its index to this list. The list stores the indices of the hits in 'trackerRecHits'
	    /* example
	       SeedingNode                     | hit index                 | hit
	       -------------------------------------------------------------------------------
	       index=  0:  [BPix1]             | hitIndicesInTree[0] (=1)  | trackerRecHits[1]
	       index=  1:   -- [BPix2]         | hitIndicesInTree[1] (=3)  | trackerRecHits[3]
	       index=  2:   --  -- [BPix3]     | hitIndicesInTree[2] (=4)  | trackerRecHits[4]
	       index=  3:   --  -- [FPix1_pos] | hitIndicesInTree[3] (=6)  | trackerRecHits[6]
	       index=  4:   --  -- [FPix1_neg] | hitIndicesInTree[4] (=7)  | trackerRecHits[7]
	 
	       The implementation has been chosen such that the tree only needs to be build once upon construction.
	    */

	    // find the first combination of hits,
	    // compatible with the seedinglayer,
	    // and with one of the tracking regions
	    std::vector<int> hitIndicesInTree(_seedingTree.numberOfNodes(),-1);
	    std::vector<unsigned int> seedHitNumbers = iterateHits(0,trackerRecHits,hitIndicesInTree,true);

	    // create a seed from those hits
	    if (seedHitNumbers.size()>1){
		// temporary hit copies to allow setting the recHitCombinationIndex
		edm::OwnVector<FastTrackerRecHit> seedHits;
		for(unsigned iIndex = 0;iIndex < seedHitNumbers.size();++iIndex){
		    seedHits.push_back(trackerRecHits[seedHitNumbers[iIndex]].hit()->clone());
		}
		fastTrackingHelper::setRecHitCombinationIndex(seedHits,icomb);

		// the actual seed creation
		seedCreator->makeSeed(*output,SeedingHitSet(&seedHits[0],&seedHits[1],
							    seedHits.size() >=3 ? &seedHits[2] : nullptr,
							    seedHits.size() >=4 ? &seedHits[3] : nullptr));
	    }
	}
    e.put(std::move(output));

}
