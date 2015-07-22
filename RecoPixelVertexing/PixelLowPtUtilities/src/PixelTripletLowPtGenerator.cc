#include "RecoPixelVertexing/PixelLowPtUtilities/interface/PixelTripletLowPtGenerator.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ThirdHitPrediction.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TripletFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/HitInfo.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

#undef Debug

using namespace std;
using namespace ctfseeding;

/*****************************************************************************/
PixelTripletLowPtGenerator::PixelTripletLowPtGenerator( const edm::ParameterSet& cfg, edm::ConsumesCollector& iC):
  HitTripletGeneratorFromPairAndLayers(), // no theMaxElement used in this class
  theTracker(nullptr),
  theClusterShapeCacheToken(iC.consumes<SiPixelClusterShapeCache>(cfg.getParameter<edm::InputTag>("clusterShapeCacheSrc")))
{
  checkMultipleScattering = cfg.getParameter<bool>("checkMultipleScattering");
  nSigMultipleScattering  = cfg.getParameter<double>("nSigMultipleScattering");
  checkClusterShape       = cfg.getParameter<bool>("checkClusterShape"); 
  rzTolerance             = cfg.getParameter<double>("rzTolerance");
  maxAngleRatio           = cfg.getParameter<double>("maxAngleRatio");
  builderName             = cfg.getParameter<string>("TTRHBuilder");
}

/*****************************************************************************/
PixelTripletLowPtGenerator::~PixelTripletLowPtGenerator() {}

/*****************************************************************************/


/*****************************************************************************/
void PixelTripletLowPtGenerator::getTracker
  (const edm::EventSetup& es)
{
  if(theTracker == 0)
  {
    // Get tracker geometry
    edm::ESHandle<TrackerGeometry> tracker;
    es.get<TrackerDigiGeometryRecord>().get(tracker);

    theTracker = tracker.product();
  }

  if(!theFilter)
  {
    theFilter = std::make_unique<TripletFilter>(es);
  }
}

/*****************************************************************************/
GlobalPoint PixelTripletLowPtGenerator::getGlobalPosition
  (const TrackingRecHit* recHit)
{
  DetId detId = recHit->geographicalId();

  return
    theTracker->idToDet(detId)->toGlobal(recHit->localPosition());
}

/*****************************************************************************/
void PixelTripletLowPtGenerator::hitTriplets(
    const TrackingRegion& region,
    OrderedHitTriplets & result,
    const edm::Event & ev,
    const edm::EventSetup& es,
    SeedingLayerSetsHits::SeedingLayerSet pairLayers,
    const std::vector<SeedingLayerSetsHits::SeedingLayer>& thirdLayers)
{

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<TrackerTopologyRcd>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();

  edm::Handle<SiPixelClusterShapeCache> clusterShapeCache;
  ev.getByToken(theClusterShapeCacheToken, clusterShapeCache);

  // Generate pairs
  OrderedHitPairs pairs; pairs.reserve(30000);
  thePairGenerator->hitPairs(region,pairs,ev,es, pairLayers);

  if (pairs.size() == 0) return;

  int size = thirdLayers.size();

  // Set aliases
  const RecHitsSortedInPhi **thirdHitMap = new const RecHitsSortedInPhi*[size]; 
  for(int il=0; il<size; il++)
    thirdHitMap[il] = &(*theLayerCache)(thirdLayers[il], region, ev, es);

  // Get tracker
  getTracker(es);

  // Look at all generated pairs
  for(OrderedHitPairs::const_iterator ip = pairs.begin();
                                      ip!= pairs.end(); ip++)
  {
    // Fill rechits and points
    vector<const TrackingRecHit*> recHits(3);
    vector<GlobalPoint> points(3);

    recHits[0] = (*ip).inner()->hit();
    recHits[1] = (*ip).outer()->hit();

#ifdef Debug
    cerr << " RecHits " + HitInfo::getInfo(*recHits[0]) +
                          HitInfo::getInfo(*recHits[1]) << endl;
#endif

    for(int i=0; i<2; i++)
      points[i] = getGlobalPosition(recHits[i]);

    // Initialize helix prediction
    ThirdHitPrediction
      thePrediction(region,
                    points[0],points[1], es,
                    nSigMultipleScattering,maxAngleRatio,builderName);

    // Look at all layers
    for(int il=0; il<size; il++)
    {
      const DetLayer * layer = thirdLayers[il].detLayer();

#ifdef Debug
      cerr << "  check layer " << layer->subDetector()
                        << " " << layer->location() << endl;
#endif

      // Get ranges for the third hit
      float phi[2],rz[2];
      thePrediction.getRanges(layer, phi,rz);

      PixelRecoRange<float> phiRange(phi[0]              , phi[1]             );
      PixelRecoRange<float>  rzRange( rz[0] - rzTolerance, rz[1] + rzTolerance);

      // Get third hit candidates from cache
      typedef RecHitsSortedInPhi::Hit Hit;
      vector<Hit> thirdHits = thirdHitMap[il]->hits(phiRange.min(),phiRange.max());
      typedef vector<Hit>::const_iterator IH;

      for (IH th=thirdHits.begin(), eh=thirdHits.end(); th < eh; ++th) 
      {
        // Fill rechit and point
        recHits[2] = (*th)->hit();
        points[2]  = getGlobalPosition(recHits[2]);

#ifdef Debug
        cerr << "  third hit " + HitInfo::getInfo(*recHits[2]) << endl;
#endif

        // Check if third hit is compatible with multiple scattering
        vector<GlobalVector> globalDirs;
        if(thePrediction.isCompatibleWithMultipleScattering
             (points[2], recHits, globalDirs, es) == false)
        {
#ifdef Debug
          cerr << "  not compatible: multiple scattering" << endl;
#endif
          if(checkMultipleScattering) continue;
        }

        // Convert to localDirs
/*
        vector<LocalVector> localDirs;
        vector<GlobalVector>::const_iterator globalDir = globalDirs.begin();
        for(vector<const TrackingRecHit *>::const_iterator
                                            recHit  = recHits.begin();
                                            recHit != recHits.end(); recHit++)
        {
          localDirs.push_back(theTracker->idToDet(
                             (*recHit)->geographicalId())->toLocal(*globalDir));
          globalDir++;
        }
*/

        // Check if the cluster shapes are compatible with thrusts
        if(checkClusterShape)
        {
          if(! theFilter->checkTrack(recHits,globalDirs,tTopo, *clusterShapeCache))
          {
#ifdef Debug
            cerr << "  not compatible: cluster shape" << endl;
#endif
            continue;
          }
        }

        // All checks passed, put triplet back
        result.push_back(OrderedHitTriplet((*ip).inner(),(*ip).outer(),*th));
      }
    }
  } 
  delete [] thirdHitMap;

  return;
}


