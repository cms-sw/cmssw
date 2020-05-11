#ifndef RecoPixelVertexingPixelTrackFittingStoreTracks_H
#define RecoPixelVertexingPixelTrackFittingStoreTracks_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

template <typename Ev, typename TWH>
void storeTracks(Ev& ev, const TWH& tracksWithHits, const TrackerTopology& ttopo) {
  auto tracks = std::make_unique<reco::TrackCollection>();
  auto recHits = std::make_unique<TrackingRecHitCollection>();
  auto trackExtras = std::make_unique<reco::TrackExtraCollection>();

  int cc = 0, nTracks = tracksWithHits.size();

  for (int i = 0; i < nTracks; i++) {
    reco::Track* track = tracksWithHits[i].first;
    const auto& hits = tracksWithHits[i].second;

    for (unsigned int k = 0; k < hits.size(); k++) {
      auto* hit = hits[k]->clone();

      track->appendHitPattern(*hit, ttopo);
      recHits->push_back(hit);
    }
    tracks->push_back(*track);
    delete track;
  }

  LogDebug("TrackProducer") << "put the collection of TrackingRecHit in the event"
                            << "\n";
  edm::OrphanHandle<TrackingRecHitCollection> ohRH = ev.put(std::move(recHits));

  edm::RefProd<TrackingRecHitCollection> hitCollProd(ohRH);
  for (int k = 0; k < nTracks; k++) {
    reco::TrackExtra theTrackExtra{};

    //fill the TrackExtra with TrackingRecHitRef
    unsigned int nHits = tracks->at(k).numberOfValidHits();
    theTrackExtra.setHits(hitCollProd, cc, nHits);
    cc += nHits;
    AlgebraicVector5 v = AlgebraicVector5(0, 0, 0, 0, 0);
    reco::TrackExtra::TrajParams trajParams(nHits, LocalTrajectoryParameters(v, 1.));
    reco::TrackExtra::Chi2sFive chi2s(nHits, 0);
    theTrackExtra.setTrajParams(std::move(trajParams), std::move(chi2s));
    trackExtras->push_back(theTrackExtra);
  }

  LogDebug("TrackProducer") << "put the collection of TrackExtra in the event"
                            << "\n";
  edm::OrphanHandle<reco::TrackExtraCollection> ohTE = ev.put(std::move(trackExtras));

  for (int k = 0; k < nTracks; k++) {
    const reco::TrackExtraRef theTrackExtraRef(ohTE, k);
    (tracks->at(k)).setExtra(theTrackExtraRef);
  }

  ev.put(std::move(tracks));
}

#endif
