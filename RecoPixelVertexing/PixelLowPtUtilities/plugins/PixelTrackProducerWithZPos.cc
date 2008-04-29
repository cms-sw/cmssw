#include "PixelTrackProducerWithZPos.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilterFactory.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrackFilter.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
//#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"


/*
class TransientTrackFromFTSFactory {
 public:

    reco::TransientTrack build (const FreeTrajectoryState & fts) const;
    reco::TransientTrack build (const FreeTrajectoryState & fts,
        const edm::ESHandle<GlobalTrackingGeometry>& trackingGeometry);
};
*/

#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"

#include <vector>
using namespace std;
using namespace reco;
using namespace pixeltrackfitting;
using namespace ctfseeding;
using edm::ParameterSet;

/*****************************************************************************/
PixelTrackProducerWithZPos::PixelTrackProducerWithZPos
  (const edm::ParameterSet& conf)
  : theConfig(conf), theFitter(0), theFilter(0), theCleaner(0), theGenerator(0), theRegionProducer(0)
{
  edm::LogInfo("PixelTrackProducerWithZPos")<<" construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}


/*****************************************************************************/
PixelTrackProducerWithZPos::~PixelTrackProducerWithZPos()
{ 
  delete theFilter;
  delete theFitter;
  delete theCleaner;
  delete theGenerator;
  delete theRegionProducer;
}

/*****************************************************************************/
void PixelTrackProducerWithZPos::beginJob(const edm::EventSetup& es)
{
  ParameterSet regfactoryPSet = theConfig.getParameter<ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet);

  ParameterSet orderedPSet = theConfig.getParameter<ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  theGenerator = OrderedHitsGeneratorFactory::get()->create( orderedName, orderedPSet);

  ParameterSet fitterPSet = theConfig.getParameter<ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);

  ParameterSet filterPSet = theConfig.getParameter<ParameterSet>("FilterPSet");
  std::string  filterName = filterPSet.getParameter<std::string>("ComponentName");
  theFilter = TrackHitsFilterFactory::get()->create( filterName, filterPSet,
es);

  ParameterSet cleanerPSet = theConfig.getParameter<ParameterSet>("CleanerPSet");
  std::string  cleanerName = cleanerPSet.getParameter<std::string>("ComponentName");
  theCleaner = PixelTrackCleanerFactory::get()->create( cleanerName, cleanerPSet);

  // Get transient track builder
  edm::ParameterSet regionPSet = regfactoryPSet.getParameter<edm::ParameterSet>("RegionPSet");
  theUseFoundVertices = regionPSet.getParameter<bool>("useFoundVertices");
  theUseChi2Cut       = regionPSet.getParameter<bool>("useChi2Cut");
  thePtMin            = regionPSet.getParameter<double>("ptMin");
  theOriginRadius     = regionPSet.getParameter<double>("originRadius");

  if(theUseFoundVertices)
  {
  edm::ESHandle<TransientTrackBuilder> builder;
  es.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  theTTBuilder = builder.product();
  }
}

/*****************************************************************************/
pair<float,float> PixelTrackProducerWithZPos::refitWithVertex
  (const reco::Track & recTrack,
   const reco::VertexCollection* vertices)
{
  TransientTrack theTransientTrack = theTTBuilder->build(recTrack);

  // If there are vertices found
  if(vertices->size() > 0)
  {
    float dzmin = -1.;
    const reco::Vertex * closestVertex = 0;

    // Look for the closest vertex in z
    for(reco::VertexCollection::const_iterator
        vertex = vertices->begin(); vertex!= vertices->end(); vertex++)
    {
      float dz = fabs(recTrack.vertex().z() - vertex->position().z());
      if(vertex == vertices->begin() || dz < dzmin)
      { dzmin = dz ; closestVertex = &(*vertex); }
    }


    // Get vertex position and error matrix
    GlobalPoint vertexPosition(closestVertex->position().x(),
                               closestVertex->position().y(),
                               closestVertex->position().z());

    float beamSize = 15e-4; // 15 um
    GlobalError vertexError(beamSize*beamSize, 0,
                            beamSize*beamSize, 0,
                            0,closestVertex->covariance(2,2));

    // Refit track with vertex constraint
    SingleTrackVertexConstraint stvc;
    pair<TransientTrack, float> result =
      stvc.constrain(theTransientTrack, vertexPosition, vertexError);

    return pair<float,float>(result.first.impactPointTSCP().pt(),
                             result.second);
  }
  else
    return pair<float,float>(recTrack.pt(), -9999);
}

/*****************************************************************************/
void PixelTrackProducerWithZPos::produce
   (edm::Event& ev, const edm::EventSetup& es)
{
  std::cerr << "["
            << theConfig.getParameter<string>("passLabel")
            << "]" << std::endl;
  
  TracksWithRecHits tracks;

  // Get vertices
  const reco::VertexCollection* vertices = 0;

  if(theUseFoundVertices)
  {
  edm::Handle<reco::VertexCollection> vertexCollection;
  ev.getByType(vertexCollection);
  vertices = vertexCollection.product();
  }

  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);

  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir)
  {
    const TrackingRegion & region = **ir;

    const OrderedSeedingHits & triplets = theGenerator->run(region,ev,es); 
    unsigned int nTriplets = triplets.size();

    std::cerr << " [TrackProducer] number of triplets     : "
              << triplets.size() << std::endl;

    // producing tracks
    for(unsigned int iTriplet = 0; iTriplet < nTriplets; ++iTriplet)
    { 
      const SeedingHitSet & triplet = triplets[iTriplet]; 
  
      std::vector<const TrackingRecHit *> hits;
      for (unsigned int iHit = 0, nHits = triplet.size(); iHit < nHits; ++iHit)
        hits.push_back( triplet[iHit] );
  
      // Fitter
      reco::Track* track = theFitter->run(es, hits, region);
  
      // Filter
      if ( ! (*theFilter)(track,hits) )
      { 
//        cerr << " [TrackProducer] track did not pass cluster shape filter" << endl;
        delete track; 
        continue; 
      }

      if(track->pt() < thePtMin ||
         track->d0() > theOriginRadius)
      {
//        cerr << " [TrackProducer] track did not pass pt and d0 filter (" << track->pt() << " " << track->d0() << ")" << endl;
        delete track; 
        continue; 
      }

      if (theUseFoundVertices)
      if (theUseChi2Cut)
      {
//        float ptv  = refitWithVertex(*track,vertices).first;
        float chi2 = refitWithVertex(*track,vertices).second;

        if(chi2/3 > 10.)
        {
//          cerr << " [TrackProducer] track did not pass chi2 filter (" << chi2 << ")" << endl;
          delete track; 
          continue; 
        }
      }
  
      // add tracks 
      tracks.push_back(TrackWithRecHits(track, hits));
    }
  }

  // Cleaner
  if(theCleaner) tracks = theCleaner->cleanTracks(tracks);

  // store tracks
  store(ev, tracks);

  // clean memory
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir)
    delete (*ir); 
}

/*****************************************************************************/
void PixelTrackProducerWithZPos::store
  (edm::Event& ev, const TracksWithRecHits & cleanedTracks)
{
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);
  std::auto_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection);
  typedef std::vector<const TrackingRecHit *> RecHits;

  int cc = 0, nTracks = cleanedTracks.size();

  for (int i = 0; i < nTracks; i++)
  {
    reco::Track* track =  cleanedTracks.at(i).first;
    const RecHits & hits = cleanedTracks.at(i).second;

    for (unsigned int k = 0; k < hits.size(); k++)
    {
      TrackingRecHit *hit = (hits.at(k))->clone();
      track->setHitPattern(*hit, k);
      recHits->push_back(hit);
    }
    tracks->push_back(*track);
    delete track;

  }

  LogDebug("TrackProducer") << "put the collection of TrackingRecHit in the event" << "\n";
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = ev.put( recHits );


  for (int k = 0; k < nTracks; k++)
  {
    reco::TrackExtra* theTrackExtra = new reco::TrackExtra();

    //fill the TrackExtra with TrackingRecHitRef
    unsigned int nHits = tracks->at(k).numberOfValidHits();
    for(unsigned int i = 0; i < nHits; ++i) {
      theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
      cc++;
    }

    trackExtras->push_back(*theTrackExtra);
    delete theTrackExtra;
  }

  LogDebug("TrackProducer") << "put the collection of TrackExtra in the event" << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = ev.put(trackExtras);

  for (int k = 0; k < nTracks; k++)
  {
    const reco::TrackExtraRef theTrackExtraRef(ohTE,k);
    (tracks->at(k)).setExtra(theTrackExtraRef);
  }

  ev.put(tracks);
}
