#include <vector>

#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "PixelTrackProducer.h"
#include "storeTracks.h"

using namespace pixeltrackfitting;
using edm::ParameterSet;

PixelTrackProducer::PixelTrackProducer(const ParameterSet& cfg) : theReconstruction(cfg, consumesCollector()) {
  edm::LogInfo("PixelTrackProducer") << " construction...";
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
}

PixelTrackProducer::~PixelTrackProducer() {}

void PixelTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("passLabel", "pixelTracks");  // What is this? It is not used anywhere in this code.
  PixelTrackReconstruction::fillDescriptions(desc);

  descriptions.add("pixelTracks", desc);
}

void PixelTrackProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  LogDebug("PixelTrackProducer, produce") << "event# :" << ev.id();

  TracksWithTTRHs tracks;
  theReconstruction.run(tracks, ev, es);
  edm::ESHandle<TrackerTopology> httopo;
  es.get<TrackerTopologyRcd>().get(httopo);

  // store tracks
  storeTracks(ev, tracks, *httopo);
}
