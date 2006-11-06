#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelHitPairTrackProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "FWCore/Framework/interface/OrphanHandle.h"
#include "RecoTracker/TkHitPairs/interface/PixelSeedLayerPairs.h"
#include "RecoTracker/TkHitPairs/interface/CombinedHitPairGenerator.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

PixelHitPairTrackProducer::PixelHitPairTrackProducer(const edm::ParameterSet& conf)
  : theConfig(conf), theFitter(0), theFilter(0), pixelLayers(0)
{
  edm::LogInfo("PixelHitPairTrackProducer")<<" construction...";
  produces<reco::TrackCollection>();
  pixelLayers =  new PixelSeedLayerPairs();
}


PixelHitPairTrackProducer::~PixelHitPairTrackProducer()
{ }


void PixelHitPairTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  typedef std::vector<const TrackingRecHit *> RecHits;

  edm::Handle<SiPixelRecHitCollection> pixelHits;
  edm::InputTag hitCollectionLabel = theConfig.getParameter<edm::InputTag>("HitCollectionLabel");
  ev.getByLabel( hitCollectionLabel, pixelHits);


  pixelLayers->init(*pixelHits,es);
  CombinedHitPairGenerator theGenerator(*pixelLayers,es);


  GlobalTrackingRegion region;
  OrderedHitPairs pairs;
  theGenerator.hitPairs(region, pairs, es);

  edm::LogInfo("PixelHitPairTrackProducer") << "number of pairs: " << pairs.size();

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

  // producing tracks

  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection);

  typedef OrderedHitPairs::const_iterator IT;
  for (IT it = pairs.begin(); it != pairs.end(); it++) {
    RecHits hits;
    hits.push_back( (*it).inner() );
    hits.push_back( (*it).outer() );
    reco::Track* track = theFitter->run(es, hits, region);
    
    if ( (*theFilter)(track) ) tracks->push_back(*track);
    delete track;
  }

  ev.put(tracks);
}

