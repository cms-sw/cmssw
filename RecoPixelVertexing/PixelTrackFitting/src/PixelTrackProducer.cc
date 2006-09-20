#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackProducer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "FWCore/Framework/interface/OrphanHandle.h"

PixelTrackProducer::PixelTrackProducer(const edm::ParameterSet& conf)
  : theConfig(conf)
{
  edm::LogInfo("PixelTrackProducer")<<" constuction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
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
  std::auto_ptr<TrackingRecHitCollection> outputRHColl(new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);

  typedef OrderedHitTriplets::const_iterator IT;
  int nTracks = 0;
  for (IT it = triplets.begin(); it != triplets.end(); it++) {
    std::vector<const TrackingRecHit *> hits;
    hits.push_back( (*it).inner() );
    hits.push_back( (*it).middle() );
    hits.push_back( (*it).outer() );
    const reco::Track *track = fitter->run(es,hits,region);
    if (track) {
      nTracks++;

      result->push_back(*track);
      delete track;

      outputRHColl->push_back( ( (*it).inner() )->clone() );
      outputRHColl->push_back( ( (*it).middle() )->clone() );
      outputRHColl->push_back( ( (*it).outer() )->clone() );
    }
  }
  cout << "nTracks" << nTracks << endl;

  LogDebug("TrackProducer") << "put the collection of TrackingRecHit in the event" << "\n";
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = ev.put( outputRHColl );

  int cc = 0;
  for (int k = 0; k < nTracks; k++)
  {
    reco::TrackExtra* theTrackExtra = new reco::TrackExtra();

    //fill the TrackExtra with TrackingRecHitRef
    for(int i = 0; i < 3; i++)
    {
      theTrackExtra->add(TrackingRecHitRef(ohRH,cc));
      cc++;
    }

    outputTEColl->push_back(*theTrackExtra);
    delete theTrackExtra;
  }

  //put the collection of TrackExtra in the event
  LogDebug("TrackProducer") << "put the collection of TrackExtra in the event" << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = ev.put(outputTEColl);

  for (int k = 0; k < nTracks; k++)
  {
    //create a TrackExtraRef
    const reco::TrackExtraRef theTrackExtraRef(ohTE,k);

    //use the TrackExtraRef to assign the TrackExtra to the Track
    (result->at(k)).setExtra(theTrackExtraRef);

  }
//


  ev.put(result);
}
