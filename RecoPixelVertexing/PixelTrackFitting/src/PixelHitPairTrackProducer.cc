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
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"

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
  ev.getByType(pixelHits);

  pixelLayers->init(*pixelHits,es);
  CombinedHitPairGenerator theGenerator(*pixelLayers,es);

//  PixelHitTripletGenerator tripGen;
//  tripGen.init(*pixelHits,es);

  GlobalTrackingRegion region;
//  OrderedHitTriplets triplets;
//  tripGen.hitTriplets(region,triplets,es);

  OrderedHitPairs pairs;
  theGenerator.hitPairs(region, pairs, es);

  edm::LogInfo("PixelHitPairTrackProducer") << "number of pairs: " << pairs.size();

  if (!theFitter) {
    std::string fitterName = theConfig.getParameter<std::string>("Fitter");
    edm::ESHandle<PixelFitter> fitterESH;
    es.get<TrackingComponentsRecord>().get(fitterName,fitterESH);
    theFitter = fitterESH.product();
  }

  if (!theFilter) {
    std::string filterName = theConfig.getParameter<std::string>("Filter");
    edm::ESHandle<PixelTrackFilter> filterESH;
    es.get<TrackingComponentsRecord>().get(filterName,filterESH);
    theFilter = filterESH.product();
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

