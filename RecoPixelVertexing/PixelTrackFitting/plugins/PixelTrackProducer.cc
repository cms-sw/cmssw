#include "PixelTrackProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include <vector>

using namespace pixeltrackfitting;
using edm::ParameterSet;

PixelTrackProducer::PixelTrackProducer(const ParameterSet& cfg)
  : runOnGPU_(cfg.getParameter<bool>("runOnGPU")),
  theReconstruction(cfg, consumesCollector()),
  theGPUReconstruction(cfg, consumesCollector())
{
  edm::LogInfo("PixelTrackProducer")<<" construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}

PixelTrackProducer::~PixelTrackProducer() { }

void PixelTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("passLabel", "pixelTracks"); // What is this? It is not used anywhere in this code.
  desc.add<bool>("runOnGPU", false);
  PixelTrackReconstruction::fillDescriptions(desc);

  descriptions.add("pixelTracks", desc);
}

void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("PixelTrackProducer, produce")<<"event# :"<<ev.id();

  TracksWithTTRHs tracks;
  if (!runOnGPU_)
    theReconstruction.run(tracks, ev, es);
  else {
    theGPUReconstruction.run(tracks, ev, es);
  }
  edm::ESHandle<TrackerTopology> httopo;
  es.get<TrackerTopologyRcd>().get(httopo);

  // store tracks
  store(ev, tracks, *httopo);
}

void PixelTrackProducer::store(edm::Event& ev, const TracksWithTTRHs& tracksWithHits, const TrackerTopology& ttopo)
{
  auto tracks = std::make_unique<reco::TrackCollection>();
  auto recHits = std::make_unique<TrackingRecHitCollection>();
  auto trackExtras = std::make_unique<reco::TrackExtraCollection>();

  int cc = 0, nTracks = tracksWithHits.size();

  for (int i = 0; i < nTracks; i++)
  {
    reco::Track* track =  tracksWithHits.at(i).first;
    const SeedingHitSet& hits = tracksWithHits.at(i).second;

    for (unsigned int k = 0; k < hits.size(); k++)
    {
      TrackingRecHit *hit = hits[k]->hit()->clone();

      track->appendHitPattern(*hit, ttopo);
      recHits->push_back(hit);
    }
    tracks->push_back(*track);
    delete track;

  }

  LogDebug("TrackProducer") << "put the collection of TrackingRecHit in the event" << "\n";
  edm::OrphanHandle <TrackingRecHitCollection> ohRH = ev.put(std::move(recHits));

  edm::RefProd<TrackingRecHitCollection> hitCollProd(ohRH);
  for (int k = 0; k < nTracks; k++)
  {
    reco::TrackExtra theTrackExtra{};

    //fill the TrackExtra with TrackingRecHitRef
    unsigned int nHits = tracks->at(k).numberOfValidHits();
    theTrackExtra.setHits(hitCollProd, cc, nHits);
    cc +=nHits;
    AlgebraicVector5 v = AlgebraicVector5(0,0,0,0,0);
    reco::TrackExtra::TrajParams trajParams(nHits,LocalTrajectoryParameters(v,1.));
    reco::TrackExtra::Chi2sFive chi2s(nHits,0);
    theTrackExtra.setTrajParams(std::move(trajParams),std::move(chi2s));
    trackExtras->push_back(theTrackExtra);
  }

  LogDebug("TrackProducer") << "put the collection of TrackExtra in the event" << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = ev.put(std::move(trackExtras));

  for (int k = 0; k < nTracks; k++)
  {
    const reco::TrackExtraRef theTrackExtraRef(ohTE,k);
    (tracks->at(k)).setExtra(theTrackExtraRef);
  }

  ev.put(std::move(tracks));

}
