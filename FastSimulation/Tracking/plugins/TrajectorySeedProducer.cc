
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
#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"

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

  /*
    testBeamspotCompatibility(false),
    beamSpot(nullptr),
    testPrimaryVertexCompatibility(false),
    primaryVertices(nullptr)
    {  */

    // The name of the TrajectorySeed Collection
    produces<TrajectorySeedCollection>();
    //Regions regions;
    const edm::ParameterSet& simTrackSelectionConfig = conf.getParameter<edm::ParameterSet>("simTrackSelection");
    // The smallest pT,dxy,dz for a simtrack
    simTrack_pTMin = simTrackSelectionConfig.getParameter<double>("pTMin");
    simTrack_maxD0 = simTrackSelectionConfig.getParameter<double>("maxD0");
    simTrack_maxZ0 = simTrackSelectionConfig.getParameter<double>("maxZ0");


    hitMasks_exists = conf.exists("hitMasks");
    if (hitMasks_exists){
      edm::InputTag hitMasksTag = conf.getParameter<edm::InputTag>("hitMasks");   
      hitMasksToken = consumes<std::vector<bool> >(hitMasksTag);
    }
    
    hitCombinationMasks_exists = conf.exists("hitCombinationMasks");
    if (hitCombinationMasks_exists){
      edm::InputTag hitCombinationMasksTag = conf.getParameter<edm::InputTag> ("hitCombinationMasks");   
      hitCombinationMasksToken = consumes<std::vector<bool> >(hitCombinationMasksTag);
    }
    /*
    std::vector<edm::InputTag> skipSimTrackTags = simTrackSelectionConfig.getParameter<std::vector<edm::InputTag> >("skipSimTrackIds");
    for ( unsigned int k=0; k<skipSimTrackTags.size(); ++k){
      skipSimTrackIdTokens.push_back(consumes<std::vector<unsigned int> >(skipSimTrackTags[k]));}
    */
    // The smallest number of hits for a track candidate
    minLayersCrossed = conf.getParameter<unsigned int>("minLayersCrossed");

    edm::InputTag beamSpotTag = conf.getParameter<edm::InputTag>("beamSpot");
    
    // The name of the hit producer
    edm::InputTag recHitTag = conf.getParameter<edm::InputTag>("recHits");
    recHitTokens = consumes<FastTMRecHitCombinations>(recHitTag);

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
    if(conf.exists("RegionFactoryPSet")){
      edm::ParameterSet regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
      std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
      theRegionProducer.reset(TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet, consumesCollector()));
      measurementTrackerEventToken = consumes<MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("MeasurementTrackerEvent"));
      const edm::ParameterSet & seedCreatorPSet = conf.getParameter<edm::ParameterSet>("SeedCreatorPSet");
      std::string seedCreatorName = seedCreatorPSet.getParameter<std::string>("ComponentName");
      seedCreator.reset(SeedCreatorFactory::get()->create( seedCreatorName, seedCreatorPSet));
    }
}
void
TrajectorySeedProducer::beginRun(edm::Run const&, const edm::EventSetup & es) 
{
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

    thePropagator = std::make_shared<PropagatorWithMaterial>(alongMomentum,0.105,magneticField);
}

bool
  TrajectorySeedProducer::pass2HitsCuts(const TrajectorySeedHitCandidate& hit1, const TrajectorySeedHitCandidate& hit2) const
{
  if(theRegionProducer){
    return testWithRegions(hit1,hit2);
  }
  else 
    {
      std::cout<<"Either region producer, or the seed creator cfg is not available."<<std::endl;
      return false;
      //The above false is just a temporary solution.
    }
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

  // the tracks to be skipped
  edm::Handle<std::vector<bool> > hitMasks;
  if (hitMasks_exists){  
    e.getByToken(hitMasksToken,hitMasks);
  }

  edm::Handle<std::vector<bool> > hitCombinationMasks;
  if (hitCombinationMasks_exists){ 
    e.getByToken(hitCombinationMasksToken,hitCombinationMasks);	
  }

    
    // SimTracks and SimVertices
  /*
    edm::Handle<edm::SimTrackContainer> theSimTracks;
    e.getByToken(simTrackToken,theSimTracks);
    
    edm::Handle<edm::SimVertexContainer> theSimVtx;
    e.getByToken(simVertexToken,theSimVtx);
  */
    edm::Handle<FastTMRecHitCombinations> recHitCombinations;
    e.getByToken(recHitTokens, recHitCombinations);

    std::auto_ptr<TrajectorySeedCollection> output{new TrajectorySeedCollection()};
    
    for ( unsigned icomb=0; icomb<recHitCombinations->size(); ++icomb)
      {
	if(hitCombinationMasks_exists
	   && icomb < hitCombinationMasks->size() 
	   && hitCombinationMasks->at(icomb))	    {
	  continue;
	}
	
	FastTMRecHitCombination recHitCombination = recHitCombinations->at(icomb);

	TrajectorySeedHitCandidate previousTrackerHit;
	TrajectorySeedHitCandidate currentTrackerHit;
	unsigned int layersCrossed=0;
	

      std::vector<TrajectorySeedHitCandidate> trackerRecHits;
      for (const auto & _hit : recHitCombination )
	{
	  if(hitMasks_exists
	     && size_t(_hit.id()) < hitMasks->size() 
	     && hitMasks->at(_hit.id()))
	    {
	      continue;
	    }
	  
	  previousTrackerHit=currentTrackerHit;
	  
	  currentTrackerHit = TrajectorySeedHitCandidate(&_hit,trackerGeometry,trackerTopology);
	  
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
      
      std::vector<int> hitIndicesInTree(_seedingTree.numberOfNodes(),-1);
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
      //Regions regions;
      regions.clear();
      if(theRegionProducer){
	es_ = &es;
	regions = theRegionProducer->regions(e,es);
	edm::Handle<MeasurementTrackerEvent> measurementTrackerEventHandle;
	e.getByToken(measurementTrackerEventToken,measurementTrackerEventHandle);
	measurementTrackerEvent = measurementTrackerEventHandle.product();
      }

      std::vector<unsigned int> seedHitNumbers = iterateHits(0,trackerRecHits,hitIndicesInTree,true);
      if (seedHitNumbers.size()>0){seedCreator->makeSeed(*output,SeedingHitSet(trackerRecHits[seedHitNumbers[0]].hit(),trackerRecHits[seedHitNumbers[1]].hit(),seedHitNumbers.size()>=3?trackerRecHits[seedHitNumbers[2]].hit():nullptr,seedHitNumbers.size()>=4?trackerRecHits[seedHitNumbers[3]].hit():nullptr));}
      }//end loop over simtracks
    e.put(output);
}

bool
  TrajectorySeedProducer::testWithRegions(const TrajectorySeedHitCandidate & innerHit,const TrajectorySeedHitCandidate & outerHit) const{
  //std::cout<<"Inside:testWithRegions1"<<std::endl;
  const DetLayer * innerLayer = measurementTrackerEvent->measurementTracker().geometricSearchTracker()->detLayer(innerHit.hit()->det()->geographicalId());
  const DetLayer * outerLayer = measurementTrackerEvent->measurementTracker().geometricSearchTracker()->detLayer(outerHit.hit()->det()->geographicalId());
  typedef PixelRecoRange<float> Range;
  //std::cout<<"Inside:testWithRegions1"<<std::endl;
  for(Regions::const_iterator ir=regions.begin(); ir < regions.end(); ++ir){
    //std::cout<<"Inside:testWithRegions1"<<std::endl;
    auto const & gs = outerHit.hit()->globalState();
    auto loc = gs.position-(*ir)->origin().basicVector();
    const HitRZCompatibility * checkRZ = (*ir)->checkRZ(innerLayer, outerHit.hit(), *es_, outerLayer,
							loc.perp(),gs.position.z(),gs.errorR,gs.errorZ);
    //std::cout<<"Inside:testWithRegions1"<<std::endl;
    float u = innerLayer->isBarrel() ? loc.perp() : gs.position.z();
    float v = innerLayer->isBarrel() ? gs.position.z() : loc.perp();
    float dv = innerLayer->isBarrel() ? gs.errorZ : gs.errorR;
    constexpr float nSigmaRZ = 3.46410161514f;
    Range allowed = checkRZ->range(u);
    float vErr = nSigmaRZ * dv;
    Range hitRZ(v-vErr, v+vErr);
    Range crossRange = allowed.intersection(hitRZ);
    //std::cout<<"Inside:testWithRegions1"<<std::endl;
    if( ! crossRange.empty()){
      //std::cout<<"Inside:testWithRegions1"<<std::endl;
      std::cout << "init seed creator"<< std::endl;
      std::cout << "ptmin: " << (**ir).ptMin() << std::endl;
      std::cout << "" << std::endl;
      seedCreator->init(**ir,*es_,0);
      std::cout << "done" << std::endl;
      return true;}
    //std::cout<<"Line485"<<std::endl;
    //seedCreator->init(**ir,*es_,0);
    //std::cout<<"Line487"<<std::endl;  
  }
  return false;
}
