#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// #include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <vector>
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoPixelVertexing/PixelTriplets/interface/PixelHitTripletGenerator.h"

PixelTrackProducer::PixelTrackProducer(const edm::ParameterSet& conf)
  : theConfig(conf)
{
  edm::LogInfo("PixelTrackProducer")<<" constuction...";
  produces<reco::TrackCollection>();
}

PixelTrackProducer::~PixelTrackProducer()
{ }

void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelTrackProducer, produce")<<"event# :"<<ev.id(); 

  edm::Handle<SiPixelRecHitCollection> pixelHits;
  ev.getByType(pixelHits);

  PixelHitTripletGenerator tripGen;
  tripGen.init(*pixelHits,es);

  GlobalTrackingRegion region;
  OrderedHitTriplets triplets;
  tripGen.hitTriplets(region,triplets,es);
  edm::LogInfo("PixelTrackProducer") << "size of triplets: "<<triplets.size();

  // get fitter
  std::string fitterName = theConfig.getParameter<std::string>("Fitter");
  edm::ESHandle<PixelFitter> fitter;
  es.get<TrackingComponentsRecord>().get(fitterName,fitter);
 

  std::auto_ptr<reco::TrackCollection> result(new reco::TrackCollection);
  
//  PixelTrackBuilderByFit builder;
  typedef OrderedHitTriplets::const_iterator IT;
  for (IT it = triplets.begin(); it != triplets.end(); it++) {
    std::vector<const TrackingRecHit *> hits;
    hits.push_back( (*it).inner() );
    hits.push_back( (*it).middle() );
    hits.push_back( (*it).outer() );
    const reco::Track *track = fitter->run(es,hits,region); 
    if (track) result->push_back(*track);
//    delete track;
  } 
  
  ev.put(result);
  LogDebug("PixelTrackProducer, produce end");
}
//DEFINE_FWK_MODULE(PixelTrackProducer)
