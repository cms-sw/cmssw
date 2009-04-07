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

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
//#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ValidHitPairFilter.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

/*
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
*/

#include <vector>
//#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
//#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/Framework/interface/ESHandle.h"

/*
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
*/


//#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"

#include <vector>
using namespace std;
using namespace reco;
using namespace pixeltrackfitting;
using namespace ctfseeding;
using edm::ParameterSet;

/*****************************************************************************/
PixelTrackProducerWithZPos::PixelTrackProducerWithZPos
  (const edm::ParameterSet& conf)
  : ps(conf), theFitter(0), theFilter(0), theHitsFilter(0), theCleaner(0), theGenerator(0), theRegionProducer(0)
{
  edm::LogInfo("PixelTrackProducerWithZPos")<<" construction...";
  produces<TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<TrackExtraCollection>();
}


/*****************************************************************************/
PixelTrackProducerWithZPos::~PixelTrackProducerWithZPos()
{ 
  delete theFilter;
  delete theHitsFilter;
  delete theFitter;
  delete theCleaner;
  delete theGenerator;
  delete theRegionProducer;
}

/*****************************************************************************/
void PixelTrackProducerWithZPos::beginJob(const edm::EventSetup& es)
{
  // Region
  ParameterSet regPSet = ps.getParameter<ParameterSet>("RegionFactoryPSet");
  string regName       = regPSet.getParameter<string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regName,regPSet);

  // Ordered hits
  ParameterSet orderedPSet = ps.getParameter<ParameterSet>("OrderedHitsFactoryPSet");
  string orderedName       = orderedPSet.getParameter<string>("ComponentName");
  theGenerator = OrderedHitsGeneratorFactory::get()->create(orderedName, orderedPSet);

  // Fitter
  ParameterSet fitterPSet = ps.getParameter<ParameterSet>("FitterPSet");
  string fitterName       = fitterPSet.getParameter<string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create(fitterName, fitterPSet);

  // Filter (ClusterShapeTrackFilter)
  ParameterSet filterPSet = ps.getParameter<ParameterSet>("FilterPSet");
  string filterName       = filterPSet.getParameter<string>("ComponentName");
  if(filterPSet.getParameter<bool>("useFilter"))
    theFilter = PixelTrackFilterWithESFactory::get()->create(filterName, filterPSet, es);

  // Filter (ValidHitPairFilter)
  theHitsFilter = PixelTrackFilterWithESFactory::get()->create("ValidHitPairFilter", filterPSet, es);

  // Cleaner
  ParameterSet cleanerPSet = ps.getParameter<ParameterSet>("CleanerPSet");
  string cleanerName       = cleanerPSet.getParameter<string>("ComponentName");
  theCleaner = PixelTrackCleanerFactory::get()->create(cleanerName, cleanerPSet);
}

/*****************************************************************************/
void PixelTrackProducerWithZPos::produce
   (edm::Event& ev, const edm::EventSetup& es)
{
  LogTrace("MinBiasTracking")
            << "\033[22;31m["
            << ps.getParameter<string>("passLabel")
            << "]\033[22;0m";
  
  TracksWithRecHits tracks;

  typedef vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);

  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir)
  {
    const TrackingRegion & region = **ir;

    const OrderedSeedingHits & triplets = theGenerator->run(region,ev,es); 
    unsigned int nTriplets = triplets.size();

    LogTrace("MinBiasTracking")
              << " [TrackProducer] number of triplets     : "
              << triplets.size();

    // Produce tracks
    for(unsigned int iTriplet = 0; iTriplet < nTriplets; ++iTriplet)
    { 
      const SeedingHitSet & triplet = triplets[iTriplet]; 
  
      vector<const TrackingRecHit *> hits;
      for (unsigned int iHit = 0, nHits = triplet.size(); iHit < nHits; ++iHit)
        hits.push_back( triplet[iHit]->hit() );
  
      // Fitter
      reco::Track* track = theFitter->run(es, hits, region);

      // Filter for pairs (ValidHitPairFilter)
      if(hits.size() == 2)
      if ( ! (*theHitsFilter)(track, hits) )
      {
        delete track; 
        continue; 
      }
  
      // Filter for triplets (ClusterShapeTrackFilter)
      if ( ! (*theFilter)(track,hits) )
      { 
        delete track; 
        continue; 
      }

      // Add tracks 
      tracks.push_back(TrackWithRecHits(track, triplet));
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
(edm::Event& ev, const TracksWithRecHits & tracksWithHits)
{
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);
  std::auto_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection);

  int cc = 0, nTracks = tracksWithHits.size();

  for (int i = 0; i < nTracks; i++)
  {
    reco::Track* track =  tracksWithHits.at(i).first;
    const SeedingHitSet& hits = tracksWithHits.at(i).second;

    for (unsigned int k = 0; k < hits.size(); k++)
    {
      TrackingRecHit *hit = hits[k]->hit()->clone();

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

