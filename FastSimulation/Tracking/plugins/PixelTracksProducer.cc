#include <memory>
#include "FastSimulation/Tracking/plugins/PixelTracksProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

//Pixel Specific stuff
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducerFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include <vector>

using namespace pixeltrackfitting;

PixelTracksProducer::PixelTracksProducer(const edm::ParameterSet& conf) : 
  theFitter(0), 
  theFilter(0), 
  theRegionProducer(0)
{  

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

  const edm::ParameterSet& regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName,
	regfactoryPSet, consumesCollector());
  
  const edm::ParameterSet& fitterPSet = conf.getParameter<edm::ParameterSet>("FitterPSet");
  std::string fitterName = fitterPSet.getParameter<std::string>("ComponentName");
  theFitter = PixelFitterFactory::get()->create( fitterName, fitterPSet);
  
  edm::ConsumesCollector iC = consumesCollector();
  const edm::ParameterSet& filterPSet = conf.getParameter<edm::ParameterSet>("FilterPSet");
  std::string filterName = filterPSet.getParameter<std::string>("ComponentName");
  theFilter = PixelTrackFilterFactory::get()->create( filterName, filterPSet, iC);
  
  // The name of the seed producer
  seedProducer = conf.getParameter<edm::InputTag>("SeedProducer");
  seedProducerToken = consumes<TrajectorySeedCollection>(seedProducer);

}

  
// Virtual destructor needed.
PixelTracksProducer::~PixelTracksProducer() {

  delete theFilter;
  delete theFitter;
  delete theRegionProducer;

} 
 

// Functions that gets called by framework every event
void 
PixelTracksProducer::produce(edm::Event& e, const edm::EventSetup& es) {        
  
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);    
  std::auto_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection);
  typedef std::vector<const TrackingRecHit *> RecHits;
  
  TracksWithRecHits pixeltracks;
  TracksWithRecHits cleanedTracks;
  
  edm::Handle<TrajectorySeedCollection> theSeeds;
  e.getByToken(seedProducerToken,theSeeds);

  // No seed -> output an empty track collection
  if(theSeeds->size() == 0) {
    e.put(tracks);
    e.put(recHits);
    e.put(trackExtras);
    return;
  }
  
  //only one region Global, but it is called at every event...
  //maybe there is a smarter way to set it only once
  //NEED TO FIX
  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(e,es);
  for (IR ir=regions.begin(), irEnd=regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion & region = **ir;
    
    // Loop over the seeds
    TrajectorySeedCollection::const_iterator aSeed = theSeeds->begin();
    TrajectorySeedCollection::const_iterator lastSeed = theSeeds->end();
    for ( ; aSeed!=lastSeed; ++aSeed ) { 
      
      // Find the first hit and last hit of the Seed
      TrajectorySeed::range theSeedingRecHitRange = aSeed->recHits();
      edm::OwnVector<TrackingRecHit>::const_iterator aSeedingRecHit = theSeedingRecHitRange.first;
      edm::OwnVector<TrackingRecHit>::const_iterator theLastSeedingRecHit = theSeedingRecHitRange.second;

      // Loop over the rechits
      std::vector<const TrackingRecHit*> TripletHits(3,static_cast<const TrackingRecHit*>(0));
      for ( unsigned i=0; aSeedingRecHit!=theLastSeedingRecHit; ++i,++aSeedingRecHit )  
	TripletHits[i] = &(*aSeedingRecHit);
      
      // fitting the triplet
      reco::Track* track = theFitter->run(es, TripletHits, region);
      
      // decide if track should be skipped according to filter 
      if ( ! (*theFilter)(track) ) { 
	delete track; 
	continue; 
      }
      
      // add tracks 
      pixeltracks.push_back(TrackWithRecHits(track, TripletHits));
      
    }
  }
  
  int cc=0;
  int nTracks = pixeltracks.size();
  for (int i = 0; i < nTracks; ++i) {

    reco::Track* track   =  pixeltracks.at(i).first;
    const RecHits & hits = pixeltracks.at(i).second;
    
    for (unsigned int k = 0; k < hits.size(); k++) {
      TrackingRecHit *hit = (hits.at(k))->clone();
      track->setHitPattern(*hit, k);
      recHits->push_back(hit);
    }

    tracks->push_back(*track);
    delete track;
    
  }
  
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = e.put( recHits );
  
  for (int k = 0; k < nTracks; ++k) {

    // reco::TrackExtra* theTrackExtra = new reco::TrackExtra();
    reco::TrackExtra theTrackExtra;
    
    //fill the TrackExtra with TrackingRecHitRef
    // unsigned int nHits = tracks->at(k).numberOfValidHits();
    unsigned nHits = 3; // We are dealing with triplets!
    for(unsigned int i = 0; i < nHits; ++i) {
      theTrackExtra.add(TrackingRecHitRef(ohRH,cc++));
      //theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
      //cc++;
    }
    
    trackExtras->push_back(theTrackExtra);
    //trackExtras->push_back(*theTrackExtra);
    //delete theTrackExtra;
  }
  
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = e.put(trackExtras);
  
  for (int k = 0; k < nTracks; k++) {

    const reco::TrackExtraRef theTrackExtraRef(ohTE,k);
    (tracks->at(k)).setExtra(theTrackExtraRef);

  }
  
  e.put(tracks);
  
  // Avoid a memory leak !
  unsigned nRegions = regions.size();
  for ( unsigned iRegions=0; iRegions<nRegions; ++iRegions ) {
    delete regions[iRegions];
  }

}

