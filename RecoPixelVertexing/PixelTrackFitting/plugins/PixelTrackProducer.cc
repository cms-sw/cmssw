#include "PixelTrackProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include <vector>

using namespace pixeltrackfitting;
using edm::ParameterSet;

PixelTrackProducer::PixelTrackProducer(const ParameterSet& cfg)
  : theReconstruction(cfg, consumesCollector())
{
  edm::LogInfo("PixelTrackProducer")<<" construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}

PixelTrackProducer::~PixelTrackProducer() { }

void PixelTrackProducer::endRun(const edm::Run &run, const edm::EventSetup& es)
{ 
  theReconstruction.halt();
}

void PixelTrackProducer::beginRun(const edm::Run &run, const edm::EventSetup& es)
{
  theReconstruction.init(es);
}

void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelTrackProducer, produce")<<"event# :"<<ev.id();

  TracksWithTTRHs tracks;
  theReconstruction.run(tracks,ev,es);

  // store tracks
  store(ev, tracks);
}

void PixelTrackProducer::store(edm::Event& ev, const TracksWithTTRHs& tracksWithHits)
{
  std::auto_ptr<reco::TrackCollection> tracks(new reco::TrackCollection());
  std::auto_ptr<TrackingRecHitCollection> recHits(new TrackingRecHitCollection());
  std::auto_ptr<reco::TrackExtraCollection> trackExtras(new reco::TrackExtraCollection());

  int cc = 0, nTracks = tracksWithHits.size();

  for (int i = 0; i < nTracks; i++)
  {
    reco::Track* track =  tracksWithHits.at(i).first;
    const SeedingHitSet& hits = tracksWithHits.at(i).second;

    for (unsigned int k = 0; k < hits.size(); k++)
    {
      TrackingRecHit *hit = hits[k]->hit()->clone();

      track->appendHitPattern(*hit);
      recHits->push_back(hit);
    }
    tracks->push_back(*track);
    delete track;

  }

  LogDebug("TrackProducer") << "put the collection of TrackingRecHit in the event" << "\n";
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = ev.put( recHits );

  edm::RefProd<TrackingRecHitCollection> hitCollProd(ohRH);
  for (int k = 0; k < nTracks; k++)
  {
    reco::TrackExtra theTrackExtra{};

    //fill the TrackExtra with TrackingRecHitRef
    unsigned int nHits = tracks->at(k).numberOfValidHits();
    theTrackExtra.setHits(hitCollProd, cc, nHits);
    cc +=nHits;
    trackExtras->push_back(theTrackExtra);
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
