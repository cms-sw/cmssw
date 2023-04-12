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

#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoTracker/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include <vector>

using namespace pixeltrackfitting;

PixelTracksProducer::PixelTracksProducer(const edm::ParameterSet& conf)
    : theRegionProducer(nullptr), ttopoToken(esConsumes()) {
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

  const edm::ParameterSet& regfactoryPSet = conf.getParameter<edm::ParameterSet>("RegionFactoryPSet");
  std::string regfactoryName = regfactoryPSet.getParameter<std::string>("ComponentName");
  theRegionProducer = TrackingRegionProducerFactory::get()->create(regfactoryName, regfactoryPSet, consumesCollector());

  fitterToken = consumes<PixelFitter>(conf.getParameter<edm::InputTag>("Fitter"));
  filterToken = consumes<PixelTrackFilter>(conf.getParameter<edm::InputTag>("Filter"));

  // The name of the seed producer
  auto seedProducer = conf.getParameter<edm::InputTag>("SeedProducer");
  seedProducerToken = consumes<TrajectorySeedCollection>(seedProducer);
}

// Virtual destructor needed.
PixelTracksProducer::~PixelTracksProducer() = default;

// Functions that gets called by framework every event
void PixelTracksProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::unique_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);
  std::unique_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection);
  std::unique_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection);
  typedef std::vector<const TrackingRecHit*> RecHits;

  TracksWithRecHits pixeltracks;
  TracksWithRecHits cleanedTracks;

  const TrackerTopology& ttopo = es.getData(ttopoToken);

  edm::Handle<PixelFitter> hfitter;
  e.getByToken(fitterToken, hfitter);
  const PixelFitter& fitter = *hfitter;

  edm::Handle<PixelTrackFilter> hfilter;
  e.getByToken(filterToken, hfilter);
  const PixelTrackFilter& theFilter = *hfilter;

  edm::Handle<TrajectorySeedCollection> theSeeds;
  e.getByToken(seedProducerToken, theSeeds);

  // No seed -> output an empty track collection
  if (theSeeds->empty()) {
    e.put(std::move(tracks));
    e.put(std::move(recHits));
    e.put(std::move(trackExtras));
    return;
  }

  //only one region Global, but it is called at every event...
  //maybe there is a smarter way to set it only once
  //NEED TO FIX
  typedef std::vector<std::unique_ptr<TrackingRegion> > Regions;
  typedef Regions::const_iterator IR;
  Regions regions = theRegionProducer->regions(e, es);
  for (IR ir = regions.begin(), irEnd = regions.end(); ir < irEnd; ++ir) {
    const TrackingRegion& region = **ir;

    // Loop over the seeds
    TrajectorySeedCollection::const_iterator aSeed = theSeeds->begin();
    TrajectorySeedCollection::const_iterator lastSeed = theSeeds->end();
    for (; aSeed != lastSeed; ++aSeed) {
      // Loop over the rechits
      std::vector<const TrackingRecHit*> TripletHits(3, static_cast<const TrackingRecHit*>(nullptr));
      unsigned int iRecHit = 0;
      for (auto const& recHit : aSeed->recHits()) {
        TripletHits[iRecHit] = &recHit;
        ++iRecHit;
      }

      // fitting the triplet
      std::unique_ptr<reco::Track> track = fitter.run(TripletHits, region);

      // decide if track should be skipped according to filter
      if (!theFilter(track.get(), TripletHits)) {
        continue;
      }

      // add tracks
      pixeltracks.push_back(TrackWithRecHits(track.release(), TripletHits));
    }
  }

  int cc = 0;
  int nTracks = pixeltracks.size();
  for (int i = 0; i < nTracks; ++i) {
    reco::Track* track = pixeltracks.at(i).first;
    const RecHits& hits = pixeltracks.at(i).second;

    for (unsigned int k = 0; k < hits.size(); k++) {
      TrackingRecHit* hit = (hits.at(k))->clone();
      track->appendHitPattern(*hit, ttopo);
      recHits->push_back(hit);
    }

    tracks->push_back(*track);
    delete track;
  }

  edm::OrphanHandle<TrackingRecHitCollection> ohRH = e.put(std::move(recHits));
  edm::RefProd<TrackingRecHitCollection> ohRHProd(ohRH);

  for (int k = 0; k < nTracks; ++k) {
    // reco::TrackExtra* theTrackExtra = new reco::TrackExtra();
    reco::TrackExtra theTrackExtra;

    //fill the TrackExtra with TrackingRecHitRef
    // unsigned int nHits = tracks->at(k).numberOfValidHits();
    const unsigned nHits = 3;  // We are dealing with triplets!
    theTrackExtra.setHits(ohRHProd, cc, nHits);
    cc += nHits;

    trackExtras->push_back(theTrackExtra);
    //trackExtras->push_back(*theTrackExtra);
    //delete theTrackExtra;
  }

  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = e.put(std::move(trackExtras));

  for (int k = 0; k < nTracks; k++) {
    const reco::TrackExtraRef theTrackExtraRef(ohTE, k);
    (tracks->at(k)).setExtra(theTrackExtraRef);
  }

  e.put(std::move(tracks));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelTracksProducer);
