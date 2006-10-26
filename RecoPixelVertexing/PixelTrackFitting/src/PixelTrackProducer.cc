#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

using namespace pixeltrackfitting;

PixelTrackProducer::PixelTrackProducer(const edm::ParameterSet& conf)
  : theConfig(conf), theFitter(0), theFilter(0), theCleaner(0), theGenerator(0)
{
  edm::LogInfo("PixelTrackProducer")<<" construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}


PixelTrackProducer::~PixelTrackProducer()
{ 
  delete theFilter;
  delete theFitter;
  delete theCleaner;
  delete theGenerator;
}


void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelTrackProducer, produce")<<"event# :"<<ev.id();
  typedef std::vector<const TrackingRecHit *> RecHits;

  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByType(pixelHits);

  if (!theGenerator) {
    edm::ParameterSet tripletsPSet = theConfig.getParameter<edm::ParameterSet>("TripletsPSet");
    theGenerator = new PixelHitTripletGenerator(tripletsPSet);
  } 
  theGenerator->init(*pixelHits,es);

  GlobalTrackingRegion region;
  OrderedHitTriplets triplets;
  theGenerator->hitTriplets(region,triplets,es);
  edm::LogInfo("PixelTrackProducer") << "number of triplets: " << triplets.size();

  if (!theFitter) {
    std::string fitterName = theConfig.getParameter<std::string>("Fitter");
    edm::ParameterSet fitterPSet = theConfig.getParameter<edm::ParameterSet>("FitterPSet");
    theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);
  }

  if (!theFilter) {
    std::string       filterName = theConfig.getParameter<std::string>("Filter");
    edm::ParameterSet filterPSet = theConfig.getParameter<edm::ParameterSet>("FilterPSet");
    theFilter = PixelTrackFilterFactory::get()->create( filterName, filterPSet);
  }

  if (!theCleaner) theCleaner = new PixelTrackCleaner();

  // producing tracks
  TracksWithRecHits tracks;

  typedef OrderedHitTriplets::const_iterator IT;
  for (IT it = triplets.begin(); it != triplets.end(); it++) {
    RecHits hits;
    hits.push_back( (*it).inner() );
    hits.push_back( (*it).middle() );
    hits.push_back( (*it).outer() );

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

  // skip ovelrapped tracks
  if(theCleaner) tracks = theCleaner->cleanTracks(tracks);

  // store tracks
  store(ev, tracks);

  //temporary
  delete theGenerator; theGenerator = 0;
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
      recHits->push_back(hit);
      track->setHitPattern(*hit, k);
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
    for(int i = 0; i < 3; i++)
    {
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
