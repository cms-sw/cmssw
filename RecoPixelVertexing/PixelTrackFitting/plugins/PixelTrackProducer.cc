#include "PixelTrackProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
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

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include <vector>

using namespace pixeltrackfitting;
using namespace ctfseeding;
using edm::ParameterSet;

PixelTrackProducer::PixelTrackProducer(const ParameterSet& cfg)
  : theConfig(cfg), theFitter(0), theFilter(0), theCleaner(0), theGenerator(0), theRegionProducer(0)
{
  edm::LogInfo("PixelTrackProducer")<<" construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}


PixelTrackProducer::~PixelTrackProducer()
{
}

void PixelTrackProducer::endRun(edm::Run &run, const edm::EventSetup& es)
{ 
  delete theFilter;
  delete theFitter;
  delete theCleaner;
  delete theGenerator;
  delete theRegionProducer;
}

void PixelTrackProducer::beginRun(edm::Run &run, const edm::EventSetup& es)
{
  ParameterSet regfactoryPSet = theConfig.getParameter<ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName,regfactoryPSet);

  ParameterSet orderedPSet =
      theConfig.getParameter<ParameterSet>("OrderedHitsFactoryPSet");
  std::string orderedName = orderedPSet.getParameter<std::string>("ComponentName");
  theGenerator = OrderedHitsGeneratorFactory::get()->create( orderedName, orderedPSet);

  ParameterSet fitterPSet = theConfig.getParameter<ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);

  ParameterSet filterPSet = theConfig.getParameter<ParameterSet>("FilterPSet");
  std::string  filterName = filterPSet.getParameter<std::string>("ComponentName");
  theFilter = PixelTrackFilterFactory::get()->create( filterName, filterPSet);

  ParameterSet cleanerPSet = theConfig.getParameter<ParameterSet>("CleanerPSet");
  std::string  cleanerName = cleanerPSet.getParameter<std::string>("ComponentName");
  theCleaner = PixelTrackCleanerFactory::get()->create( cleanerName, cleanerPSet);

}

void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelTrackProducer, produce")<<"event# :"<<ev.id();

  TracksWithRecHits tracks;

  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(ev,es);

  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;

    const OrderedSeedingHits & triplets =  theGenerator->run(region,ev,es); 
    unsigned int nTriplets = triplets.size();
    edm::LogInfo("PixelTrackProducer") << "number of proto tracks: " << triplets.size();

    // producing tracks
    for (unsigned int iTriplet = 0; iTriplet < nTriplets; ++iTriplet) { 
      const SeedingHitSet & triplet = triplets[iTriplet]; 

      std::vector<const TrackingRecHit *> hits;
      for (unsigned int iHit = 0, nHits = triplet.size(); iHit < nHits; ++iHit) {
        hits.push_back( triplet[iHit] );
      }

      // fitting
      reco::Track* track = theFitter->run(es, hits, region);

      // decide if track should be skipped according to filter 
      if ( ! (*theFilter)(track) ) { 
        delete track; 
        continue; 
      }

      // add tracks 
      tracks.push_back(TrackWithRecHits(track, hits));
    }
  }

  // skip ovelrapped tracks
  if(theCleaner) tracks = theCleaner->cleanTracks(tracks);

  // store tracks
  store(ev, tracks);

  // clean memory
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) delete (*ir); 
}

void PixelTrackProducer::store(edm::Event& ev, const TracksWithRecHits & cleanedTracks)
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
